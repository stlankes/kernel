//! Driver for the Interrupt Translation Service (ITS) of the GICv3, which
//! translates message-signaled interrupts of devices into LPIs.

use alloc::collections::BTreeMap;
use core::arch::asm;

use align_address::Align;
use free_list::PageLayout;
use hermit_sync::{InterruptTicketMutex, OnceCell};
use memory_addresses::{PhysAddr, VirtAddr};

use crate::arch::mm::paging::{self, BasePageSize, PageSize, PageTableEntryFlags};
use crate::mm::device_alloc::DeviceAlloc;
use crate::mm::{FrameAlloc, PageAlloc, PageRangeAllocator};

/// INTID of the first LPI
pub(crate) const LPI_BASE: u32 = 8192;
/// Number of INTID bits, which allows the LPIs `LPI_BASE..2 * LPI_BASE`
const ID_BITS: u32 = 14;
/// Priority of all LPIs, group 1 and enabled
const LPI_CONFIG: u8 = 0xa0 | (1 << 1) | 1;
/// Size of the command queue
const CMD_QUEUE_SIZE: usize = 0x1000;
/// Upper limit of the DeviceID bits, which the device table supports
const MAX_DEVICE_BITS: u32 = 16;

// Registers of a redistributor (`RD_base` frame)
const GICR_CTLR: usize = 0x0;
const GICR_TYPER: usize = 0x8;
const GICR_PROPBASER: usize = 0x70;
const GICR_PENDBASER: usize = 0x78;

// Registers of the ITS
const GITS_CTLR: usize = 0x0;
const GITS_TYPER: usize = 0x8;
const GITS_CBASER: usize = 0x80;
const GITS_CWRITER: usize = 0x88;
const GITS_CREADR: usize = 0x90;
const GITS_BASER: usize = 0x100;
const GITS_TRANSLATER: u64 = 0x1_0040;

// Memory attributes: inner shareable, write-back cacheable (read and write allocate)
const SHAREABILITY_INNER: u64 = 0b01;
const CACHE_WB: u64 = 0b111;
const CACHE_NC: u64 = 0b001;

/// Location of the redistributors
struct Redistributors {
	virt: VirtAddr,
	phys: PhysAddr,
	stride: usize,
}

impl Redistributors {
	fn rd_base(&self, cpu: usize) -> VirtAddr {
		self.virt + (cpu * self.stride) as u64
	}
}

static REDISTRIBUTORS: OnceCell<Redistributors> = OnceCell::new();
/// Physical address of the LPI configuration table, which all redistributors share
static LPI_CONFIG_TABLE: OnceCell<PhysAddr> = OnceCell::new();
static ITS: InterruptTicketMutex<Option<Its>> = InterruptTicketMutex::new(None);

unsafe fn read64(addr: VirtAddr) -> u64 {
	u64::from_le(unsafe { addr.as_ptr::<u64>().read_volatile() })
}

unsafe fn write64(addr: VirtAddr, value: u64) {
	unsafe { addr.as_mut_ptr::<u64>().write_volatile(value.to_le()) }
}

unsafe fn read32(addr: VirtAddr) -> u32 {
	u32::from_le(unsafe { addr.as_ptr::<u32>().read_volatile() })
}

unsafe fn write32(addr: VirtAddr, value: u32) {
	unsafe { addr.as_mut_ptr::<u32>().write_volatile(value.to_le()) }
}

/// Writes the data cache lines of `len` bytes at `ptr` back to memory, so that
/// the GIC observes them even if it does not snoop the caches.
fn clean_dcache(ptr: *const u8, len: usize) {
	let ctr: u64;
	// SAFETY: Reading CTR_EL0 has no side effects.
	unsafe {
		asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags));
	}
	// CTR_EL0.DminLine is the log2 of the number of words of the smallest cache line.
	let line = 4 << ((ctr >> 16) & 0xf);
	let start = ptr.addr().align_down(line);
	for addr in (start..ptr.addr() + len).step_by(line) {
		// SAFETY: Cleaning a cache line does not change memory.
		unsafe {
			asm!("dc cvac, {}", in(reg) addr, options(nostack, preserves_flags));
		}
	}
	// SAFETY: A barrier does not change memory.
	unsafe {
		asm!("dsb sy", options(nostack, preserves_flags));
	}
}

/// Allocates zeroed and physically contiguous memory.
fn alloc_zeroed(size: usize, align: usize) -> PhysAddr {
	let size = size.align_up(BasePageSize::SIZE as usize);
	let layout = PageLayout::from_size_align(size, align).unwrap();
	let range = FrameAlloc::allocate(layout).unwrap();
	let phys = PhysAddr::from(range.start());
	let ptr = DeviceAlloc.ptr_from::<u8>(phys);
	unsafe {
		ptr.write_bytes(0, size);
	}
	clean_dcache(ptr, size);
	phys
}

/// Returns `reg`, in which the field at `shift` with `mask` is replaced by `value`.
fn with_field(reg: u64, shift: u32, mask: u64, value: u64) -> u64 {
	(reg & !(mask << shift)) | ((value & mask) << shift)
}

/// Remembers the location of the redistributors and enables the LPIs of the boot processor.
pub(crate) fn init(gicr_virt: VirtAddr, gicr_phys: PhysAddr) {
	// GICR_TYPER.VLPIS: a redistributor of GICv4 has two additional frames.
	let vlpis = unsafe { read64(gicr_virt + GICR_TYPER as u64) } & (1 << 1) != 0;
	let stride = if vlpis { 0x40000 } else { 0x20000 };
	REDISTRIBUTORS
		.set(Redistributors {
			virt: gicr_virt,
			phys: gicr_phys,
			stride,
		})
		.ok()
		.unwrap();

	let config_size = (1usize << ID_BITS) - LPI_BASE as usize;
	LPI_CONFIG_TABLE
		.set(alloc_zeroed(config_size, BasePageSize::SIZE as usize))
		.unwrap();

	init_cpu(0);
}

/// Enables the LPIs of the redistributor of `cpu`.
pub(crate) fn init_cpu(cpu: usize) {
	let Some(redists) = REDISTRIBUTORS.get() else {
		return;
	};
	let rd_base = redists.rd_base(cpu);

	// GICR_TYPER.PLPIS
	if unsafe { read64(rd_base + GICR_TYPER as u64) } & 1 == 0 {
		warn!("Redistributor of core {cpu} does not support LPIs");
		return;
	}

	let config_table = *LPI_CONFIG_TABLE.get().unwrap();
	let mut propbaser = config_table.as_u64() | u64::from(ID_BITS - 1);
	propbaser = with_field(propbaser, 7, 0b111, CACHE_WB);
	propbaser = with_field(propbaser, 10, 0b11, SHAREABILITY_INNER);
	unsafe {
		write64(rd_base + GICR_PROPBASER as u64, propbaser);
	}
	if (unsafe { read64(rd_base + GICR_PROPBASER as u64) } >> 10) & 0b11 == 0 {
		propbaser = with_field(propbaser, 7, 0b111, CACHE_NC);
		propbaser = with_field(propbaser, 10, 0b11, 0);
		unsafe {
			write64(rd_base + GICR_PROPBASER as u64, propbaser);
		}
	}

	// The pending table has one bit per INTID and has to be aligned to 64 KiB.
	let pending_table = alloc_zeroed((1 << ID_BITS) / 8, 0x10000);
	let mut pendbaser = pending_table.as_u64() | (1 << 62);
	pendbaser = with_field(pendbaser, 7, 0b111, CACHE_WB);
	pendbaser = with_field(pendbaser, 10, 0b11, SHAREABILITY_INNER);
	unsafe {
		write64(rd_base + GICR_PENDBASER as u64, pendbaser);
	}
	if (unsafe { read64(rd_base + GICR_PENDBASER as u64) } >> 10) & 0b11 == 0 {
		pendbaser = with_field(pendbaser, 7, 0b111, CACHE_NC);
		pendbaser = with_field(pendbaser, 10, 0b11, 0);
		unsafe {
			write64(rd_base + GICR_PENDBASER as u64, pendbaser);
		}
	}

	unsafe {
		asm!("dsb sy", options(nostack, preserves_flags));
		let ctlr = read32(rd_base + GICR_CTLR as u64);
		write32(rd_base + GICR_CTLR as u64, ctlr | 1);
		asm!("dsb sy", options(nostack, preserves_flags));
	}

	if let Some(its) = ITS.lock().as_mut() {
		its.map_collection(cpu);
	}
}

/// Initializes the ITS at `phys` with `size` bytes.
pub(crate) fn init_its(phys: PhysAddr, size: usize, num_cpus: usize) {
	let layout = PageLayout::from_size_align(size, 0x10000).unwrap();
	let page_range = PageAlloc::allocate(layout).unwrap();
	let virt = VirtAddr::from(page_range.start());
	paging::map::<BasePageSize>(
		virt,
		phys,
		size / BasePageSize::SIZE as usize,
		PageTableEntryFlags::device(),
	);

	let mut its = Its::new(virt, phys, num_cpus);
	its.map_collection(0);
	*ITS.lock() = Some(its);
}

/// Returns the physical address of the register, to which devices write their messages.
pub(crate) fn translater() -> Option<PhysAddr> {
	ITS.lock().as_ref().map(|its| its.translater)
}

/// Maps the events `0..num_events` of the device `device_id` to the LPIs
/// `first_lpi..first_lpi + num_events`, which are delivered to the boot processor.
pub(crate) fn map_device(device_id: u32, num_events: u32, first_lpi: u32) {
	assert!(
		first_lpi + num_events <= (1 << ID_BITS) - LPI_BASE,
		"Out of LPIs"
	);
	let mut guard = ITS.lock();
	let its = guard.as_mut().unwrap();
	its.map_device(device_id, num_events);
	for event in 0..num_events {
		let lpi = first_lpi + event;
		set_lpi_config(lpi, LPI_CONFIG);
		its.command([
			0x0a | (u64::from(device_id) << 32),
			u64::from(event) | (u64::from(LPI_BASE + lpi) << 32),
			0,
			0,
		]);
		its.command([0x0c | (u64::from(device_id) << 32), u64::from(event), 0, 0]);
	}
	its.sync(0);
}

fn set_lpi_config(lpi: u32, config: u8) {
	let config_table = *LPI_CONFIG_TABLE.get().unwrap();
	let entry = DeviceAlloc.ptr_from::<u8>(config_table + u64::from(lpi));
	unsafe {
		entry.write_volatile(config);
	}
	clean_dcache(entry, 1);
}

struct Its {
	base: VirtAddr,
	translater: PhysAddr,
	cmd_queue: *mut u8,
	cmd_write: usize,
	/// Size of an entry of an interrupt translation table
	itt_entry_size: usize,
	/// Whether the target of a collection is the physical address of the redistributor
	pta: bool,
	/// Number of DeviceIDs, which the device table supports
	num_devices: u32,
	devices: BTreeMap<u32, PhysAddr>,
}

// SAFETY: The command queue is only accessed with the lock of `ITS`.
unsafe impl Send for Its {}

impl Its {
	fn new(base: VirtAddr, phys: PhysAddr, num_cpus: usize) -> Self {
		unsafe {
			let ctlr = read32(base + GITS_CTLR as u64);
			write32(base + GITS_CTLR as u64, ctlr & !1);
			// Wait until the ITS is quiescent.
			while read32(base + GITS_CTLR as u64) & (1 << 31) == 0 {
				core::hint::spin_loop();
			}
		}

		let typer = unsafe { read64(base + GITS_TYPER as u64) };
		let itt_entry_size = usize::try_from(((typer >> 4) & 0xf) + 1).unwrap();
		let device_bits = u32::try_from(((typer >> 13) & 0x1f) + 1).unwrap();
		let pta = typer & (1 << 19) != 0;

		let cmd_phys = alloc_zeroed(CMD_QUEUE_SIZE, 0x10000);
		let mut cbaser = cmd_phys.as_u64() | (1 << 63);
		cbaser |= ((CMD_QUEUE_SIZE / 0x1000) - 1) as u64;
		cbaser = with_field(cbaser, 59, 0b111, CACHE_WB);
		cbaser = with_field(cbaser, 10, 0b11, SHAREABILITY_INNER);
		unsafe {
			write64(base + GITS_CBASER as u64, cbaser);
			if (read64(base + GITS_CBASER as u64) >> 10) & 0b11 == 0 {
				cbaser = with_field(cbaser, 59, 0b111, CACHE_NC);
				cbaser = with_field(cbaser, 10, 0b11, 0);
				write64(base + GITS_CBASER as u64, cbaser);
			}
			write64(base + GITS_CWRITER as u64, 0);
		}

		let num_devices = 1 << device_bits.min(MAX_DEVICE_BITS);
		for i in 0..8 {
			let reg = base + (GITS_BASER + 8 * i) as u64;
			let baser = unsafe { read64(reg) };
			let entries = match (baser >> 56) & 0b111 {
				// Devices
				1 => num_devices as usize,
				// Collections
				4 => num_cpus,
				_ => continue,
			};
			let entry_size = usize::try_from(((baser >> 48) & 0x1f) + 1).unwrap();
			Self::alloc_table(reg, entries * entry_size);
		}

		unsafe {
			let ctlr = read32(base + GITS_CTLR as u64);
			write32(base + GITS_CTLR as u64, ctlr | 1);
		}

		info!(
			"Initialized ITS at {phys:p}: {device_bits} DeviceID bits, ITT entry size {itt_entry_size}, PTA {pta}"
		);

		Self {
			base,
			translater: phys + GITS_TRANSLATER,
			cmd_queue: DeviceAlloc.ptr_from(cmd_phys),
			cmd_write: 0,
			itt_entry_size,
			pta,
			num_devices,
			devices: BTreeMap::new(),
		}
	}

	/// Allocates a flat table with `size` bytes for `GITS_BASER<n>` at `reg`.
	fn alloc_table(reg: VirtAddr, size: usize) {
		// Try 4 KiB pages first and use the page size, which the ITS accepts.
		let mut baser = unsafe { read64(reg) };
		baser = with_field(baser, 8, 0b11, 0b00);
		baser = with_field(baser, 59, 0b111, CACHE_WB);
		baser = with_field(baser, 10, 0b11, SHAREABILITY_INNER);
		unsafe {
			write64(reg, baser);
			baser = read64(reg);
		}
		let page_size = match (baser >> 8) & 0b11 {
			0b00 => 0x1000,
			0b01 => 0x4000,
			_ => 0x10000,
		};
		if (baser >> 10) & 0b11 == 0 {
			baser = with_field(baser, 59, 0b111, CACHE_NC);
		}

		let size = size.align_up(page_size);
		let pages = size / page_size;
		assert!(pages <= 256);
		let table = alloc_zeroed(size, page_size);
		baser = with_field(baser, 12, (1 << 36) - 1, table.as_u64() >> 12);
		baser = with_field(baser, 0, 0xff, (pages - 1) as u64);
		baser |= 1 << 63;
		unsafe {
			write64(reg, baser);
		}
	}

	/// Executes `cmd` and waits for its completion.
	fn command(&mut self, cmd: [u64; 4]) {
		let slot = self.cmd_queue.wrapping_add(self.cmd_write).cast::<u64>();
		for (i, dword) in cmd.iter().enumerate() {
			unsafe {
				slot.add(i).write_volatile(dword.to_le());
			}
		}
		clean_dcache(slot.cast(), 32);

		self.cmd_write = (self.cmd_write + 32) % CMD_QUEUE_SIZE;
		unsafe {
			write64(self.base + GITS_CWRITER as u64, self.cmd_write as u64);
			loop {
				let creadr = read64(self.base + GITS_CREADR as u64);
				// GITS_CREADR.Stalled
				assert!(creadr & 1 == 0, "ITS command queue stalled");
				if creadr as usize == self.cmd_write {
					break;
				}
				core::hint::spin_loop();
			}
		}
	}

	/// Returns the encoding of the redistributor of `cpu` in commands.
	fn rd_base(&self, cpu: usize) -> u64 {
		let redists = REDISTRIBUTORS.get().unwrap();
		if self.pta {
			(redists.phys.as_u64() + (cpu * redists.stride) as u64) & !0xffff
		} else {
			// GICR_TYPER.Processor_Number
			let typer = unsafe { read64(redists.rd_base(cpu) + GICR_TYPER as u64) };
			((typer >> 8) & 0xffff) << 16
		}
	}

	fn sync(&mut self, cpu: usize) {
		let rd_base = self.rd_base(cpu);
		self.command([0x05, 0, rd_base, 0]);
	}

	/// Maps the collection `cpu` to the redistributor of `cpu`.
	fn map_collection(&mut self, cpu: usize) {
		let rd_base = self.rd_base(cpu);
		self.command([0x09, 0, (1 << 63) | rd_base | cpu as u64, 0]);
		self.command([0x0d, 0, cpu as u64, 0]);
		self.sync(cpu);
	}

	/// Maps `device_id` to a new interrupt translation table with `num_events` entries.
	fn map_device(&mut self, device_id: u32, num_events: u32) {
		assert!(device_id < self.num_devices);
		assert!(!self.devices.contains_key(&device_id));
		let event_bits = num_events.next_power_of_two().trailing_zeros().max(1);
		let itt = alloc_zeroed(
			(1 << event_bits) * self.itt_entry_size,
			BasePageSize::SIZE as usize,
		);
		self.command([
			0x08 | (u64::from(device_id) << 32),
			u64::from(event_bits - 1),
			(1 << 63) | (itt.as_u64() & !0xff),
			0,
		]);
		self.devices.insert(device_id, itt);
	}
}
