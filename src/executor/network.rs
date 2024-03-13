use alloc::boxed::Box;
use core::future;
use core::ops::DerefMut;
use core::sync::atomic::{AtomicU16, Ordering};
use core::task::Poll;

use hermit_sync::InterruptTicketMutex;
use smoltcp::iface::{SocketHandle, SocketSet};
#[cfg(feature = "dhcpv4")]
use smoltcp::socket::dhcpv4;
#[cfg(feature = "tcp")]
use smoltcp::socket::tcp;
#[cfg(feature = "udp")]
use smoltcp::socket::udp;
use smoltcp::socket::AnySocket;
use smoltcp::time::{Duration, Instant};
#[cfg(feature = "dhcpv4")]
use smoltcp::wire::{IpCidr, Ipv4Address, Ipv4Cidr};

use crate::arch;
use crate::executor::device::HermitNet;
use crate::executor::spawn;
use crate::scheduler::PerCoreSchedulerExt;

pub(crate) enum NetworkState<'a> {
	Missing,
	InitializationFailed,
	Initialized(Box<NetworkInterface<'a>>),
}

impl<'a> NetworkState<'a> {
	pub fn as_nic_mut(&mut self) -> Result<&mut NetworkInterface<'a>, &'static str> {
		match self {
			NetworkState::Initialized(nic) => Ok(nic),
			_ => Err("Network is not initialized!"),
		}
	}
}

pub(crate) type Handle = SocketHandle;

static LOCAL_ENDPOINT: AtomicU16 = AtomicU16::new(0);
pub(crate) static NIC: InterruptTicketMutex<NetworkState<'_>> =
	InterruptTicketMutex::new(NetworkState::Missing);

pub(crate) struct NetworkInterface<'a> {
	pub(super) iface: smoltcp::iface::Interface,
	pub(super) sockets: SocketSet<'a>,
	pub(super) device: HermitNet,
	#[cfg(feature = "dhcpv4")]
	pub(super) dhcp_handle: SocketHandle,
}

#[cfg(target_arch = "x86_64")]
fn start_endpoint() -> u16 {
	((unsafe { core::arch::x86_64::_rdtsc() }) % (u16::MAX as u64))
		.try_into()
		.unwrap()
}

#[cfg(target_arch = "aarch64")]
fn start_endpoint() -> u16 {
	use core::arch::asm;
	let value: u64;

	unsafe {
		asm!(
			"mrs {value}, cntpct_el0",
			value = out(reg) value,
			options(nostack),
		);
	}

	(value % (u16::MAX as u64)).try_into().unwrap()
}

#[cfg(target_arch = "riscv64")]
fn start_endpoint() -> u16 {
	(riscv::register::time::read64() % (u16::MAX as u64))
		.try_into()
		.unwrap()
}

#[inline]
pub(crate) fn now() -> Instant {
	Instant::from_micros_const(arch::kernel::systemtime::now_micros().try_into().unwrap())
}

async fn network_run() {
	future::poll_fn(|_cx| {
		if let Some(mut guard) = NIC.try_lock() {
			match guard.deref_mut() {
				NetworkState::Initialized(nic) => {
					nic.poll_common(now());
					Poll::Pending
				}
				_ => Poll::Ready(()),
			}
		} else {
			// another task is already using the NIC => don't check
			Poll::Pending
		}
	})
	.await
}

pub(crate) fn init() {
	info!("Try to initialize network!");

	// initialize variable, which contains the next local endpoint
	LOCAL_ENDPOINT.store(start_endpoint(), Ordering::Relaxed);

	let mut guard = NIC.lock();

	*guard = NetworkInterface::create();

	if let NetworkState::Initialized(nic) = guard.deref_mut() {
		let time = now();
		nic.poll_common(time);
		let wakeup_time = nic
			.poll_delay(time)
			.map(|d| crate::arch::processor::get_timer_ticks() + d.total_micros());
		crate::core_scheduler().add_network_timer(wakeup_time);

		spawn(network_run());
	}
}

impl<'a> NetworkInterface<'a> {
	#[cfg(feature = "udp")]
	pub(crate) fn create_udp_handle(&mut self) -> Result<Handle, ()> {
		let udp_rx_buffer =
			udp::PacketBuffer::new(vec![udp::PacketMetadata::EMPTY; 4], vec![0; 65535]);
		let udp_tx_buffer =
			udp::PacketBuffer::new(vec![udp::PacketMetadata::EMPTY; 4], vec![0; 65535]);
		let udp_socket = udp::Socket::new(udp_rx_buffer, udp_tx_buffer);
		let udp_handle = self.sockets.add(udp_socket);

		Ok(udp_handle)
	}

	#[cfg(feature = "tcp")]
	pub(crate) fn create_tcp_handle(&mut self) -> Result<Handle, ()> {
		let tcp_rx_buffer = tcp::SocketBuffer::new(vec![0; 65535]);
		let tcp_tx_buffer = tcp::SocketBuffer::new(vec![0; 65535]);
		let mut tcp_socket = tcp::Socket::new(tcp_rx_buffer, tcp_tx_buffer);
		tcp_socket.set_nagle_enabled(true);
		let tcp_handle = self.sockets.add(tcp_socket);

		Ok(tcp_handle)
	}

	pub(crate) fn poll_common(&mut self, timestamp: Instant) {
		let _ = self
			.iface
			.poll(timestamp, &mut self.device, &mut self.sockets);

		#[cfg(feature = "dhcpv4")]
		match self
			.sockets
			.get_mut::<dhcpv4::Socket<'_>>(self.dhcp_handle)
			.poll()
		{
			None => {}
			Some(dhcpv4::Event::Configured(config)) => {
				info!("DHCP config acquired!");
				info!("IP address:      {}", config.address);
				self.iface.update_ip_addrs(|addrs| {
					if let Some(dest) = addrs.iter_mut().next() {
						*dest = IpCidr::Ipv4(config.address);
					} else if addrs.push(IpCidr::Ipv4(config.address)).is_err() {
						info!("Unable to update IP address");
					}
				});
				if let Some(router) = config.router {
					info!("Default gateway: {}", router);
					self.iface
						.routes_mut()
						.add_default_ipv4_route(router)
						.unwrap();
				} else {
					info!("Default gateway: None");
					self.iface.routes_mut().remove_default_ipv4_route();
				}

				for (i, s) in config.dns_servers.iter().enumerate() {
					info!("DNS server {}:    {}", i, s);
				}
			}
			Some(dhcpv4::Event::Deconfigured) => {
				info!("DHCP lost config!");
				let cidr = Ipv4Cidr::new(Ipv4Address::UNSPECIFIED, 0);
				self.iface.update_ip_addrs(|addrs| {
					if let Some(dest) = addrs.iter_mut().next() {
						*dest = IpCidr::Ipv4(cidr);
					}
				});
				self.iface.routes_mut().remove_default_ipv4_route();
			}
		};
	}

	pub(crate) fn poll_delay(&mut self, timestamp: Instant) -> Option<Duration> {
		self.iface.poll_delay(timestamp, &self.sockets)
	}

	#[allow(dead_code)]
	pub(crate) fn get_socket<T: AnySocket<'a>>(&self, handle: SocketHandle) -> &T {
		self.sockets.get(handle)
	}

	pub(crate) fn get_mut_socket<T: AnySocket<'a>>(&mut self, handle: SocketHandle) -> &mut T {
		self.sockets.get_mut(handle)
	}

	pub(crate) fn get_socket_and_context<T: AnySocket<'a>>(
		&mut self,
		handle: SocketHandle,
	) -> (&mut T, &mut smoltcp::iface::Context) {
		(self.sockets.get_mut(handle), self.iface.context())
	}

	pub(crate) fn destroy_socket(&mut self, handle: Handle) {
		// This deallocates the socket's buffers
		self.sockets.remove(handle);
	}

	#[cfg(feature = "shell")]
	pub(crate) fn print_config(&self) {
		use smoltcp::phy::Device;
		use smoltcp::wire::EthernetAddress;
		use smoltcp::wire::HardwareAddress::Ethernet;
		use smoltcp::wire::IpCidr::{Ipv4, Ipv6};

		if let Ethernet(EthernetAddress(hardware_addr)) = self.iface.hardware_addr() {
			println!(
				"MAC address: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
				hardware_addr[0],
				hardware_addr[1],
				hardware_addr[2],
				hardware_addr[3],
				hardware_addr[4],
				hardware_addr[5]
			);
		}

		for i in self.iface.ip_addrs() {
			match *i {
				Ipv4(ip) => {
					if let smoltcp::wire::Ipv4Address(ip) = ip.address() {
						println!("IPv4: {}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]);
					}
				}
				Ipv6(ip) => {
					if let smoltcp::wire::Ipv6Address(ip) = ip.address() {
						println!(
							"IPv6: {:04x}.{:04x}.{:04x}.{:04x}:{:04x}.{:04x}.{:04x}.{:04x}",
							ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7]
						);
					}
				}
			}
		}

		println!("MTU: {}", self.device.capabilities().max_transmission_unit);
	}
}

#[cfg(feature = "shell")]
pub(crate) fn print_network_configuration() {
	if let Ok(nic) = crate::executor::network::NIC.lock().as_nic_mut() {
		nic.print_config();
	}
}

#[inline]
pub(crate) fn network_delay(timestamp: Instant) -> Option<Duration> {
	crate::executor::network::NIC
		.lock()
		.as_nic_mut()
		.unwrap()
		.poll_delay(timestamp)
}

#[inline]
fn network_poll(timestamp: Instant) {
	crate::executor::network::NIC
		.lock()
		.as_nic_mut()
		.unwrap()
		.poll_common(timestamp);
}
