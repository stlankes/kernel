// Copyright (c) 2020 Frederik Schulz, RWTH Aachen University
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This module contains Virtio's packed virtqueue. 
//! See Virito specification v1.1. - 2.7
use alloc::vec::Vec;

use super::super::transport::pci::ComCfg;
use super::{VqSize, VqIndex, MemPool, MemDescrId, MemDescr, BufferToken, TransferToken, Transfer, TransferState, Buffer, BuffSpec, Bytes, AsSliceU8, Pinned, Virtq, DescrFlags};
use super::error::VirtqError;
use self::error::VqPackedError;
use core::convert::TryFrom;
use alloc::boxed::Box;
use core::cell::RefCell;
use core::sync::atomic::{fence, Ordering};
use alloc::rc::Rc;
use core::ops::Deref;

/// A newtype of bool used for convenience in context with 
/// packed queues wrap counter.
///
/// For more details see Virtio specification v1.1. - 2.7.1
#[derive(Copy, Clone, Debug)]
struct WrapCount(bool);

impl WrapCount {
    /// Returns a new WrapCount struct initalized to true or 1.
    /// 
    /// See virtio specification v1.1. - 2.7.1
    fn new() -> Self {
        WrapCount(true)
    }

    /// Toogles a given wrap count to respectiver other value.
    ///
    /// If WrapCount(true) returns WrapCount(false), 
    /// if WrapCount(false) returns WrapCount(true).
    fn wrap(&mut self) {
        if self.0 == false {
            self.0 = true;
        } else {
            self.0 = false;
        }
    }

    /// Creates avail and used flags inside u16 in accordance to the 
    /// virito specification v1.1. - 2.7.1
    ///
    /// I.e.: Set avail flag to match the WrapCount and the used flag
    /// to NOT match the WrapCount.
    fn as_flags(&self) -> u16 {
        if self.0 == true {
            1 << 7
        } else {
            1 << 15
        }
    } 
}

/// Structure which allows to control raw ring and operate easily on it
/// 
/// WARN: NEVER PUSH TO THE RING AFTER DESCRIPTORRING HAS BEEN INITALIZED AS THIS WILL PROBABLY RESULT IN A 
/// RELOCATION OF THE VECTOR AND HENCE THE DEVICE WILL NO LONGER NO THE RINGS ADDRESS!
struct DescriptorRing {
    ring: Box<[Descriptor]>,
    //ring: Pinned<Vec<Descriptor>>, 
    tkn_ref_ring: Box<[*mut TransferToken]>,

    // Controlling variables for the ring
    //
    /// where to insert availble descriptors next
    write_index: usize,
    /// How much descriptors can be inserted
    capacity: usize,
    /// Where to expect the next used descriptor by the device
    poll_index: usize,
    /// See Virtio specification v1.1. - 2.7.1
    wrap_count: WrapCount,
}

impl DescriptorRing {
    fn new(size: u16) -> Self {
        let size = usize::try_from(size).unwrap();
        // WARN: Uncatched as usize call here. Could panic if used with usize < u16
        let mut ring = Box::new(Vec::with_capacity(size));
        for _ in 0..size {
            ring.push(Descriptor {
                address: 0,
                len: 0,
                buff_id: 0,
                flags: 0,
            });
        }
        
        // Descriptor ID's run from 1 to size_of_queue. In order to index directly into the 
        // refernece ring via an ID it is much easier to simply have an array of size = size_of_queue + 1
        // and do not care about the first element beeing unused.
        let tkn_ref_ring = vec![0usize as *mut TransferToken; size+1].into_boxed_slice();

        DescriptorRing { 
            ring: ring.into_boxed_slice(),
            tkn_ref_ring,
            write_index: 0,
            capacity: size,
            poll_index: 0,
            wrap_count: WrapCount::new(),
         }
    }

    /// # Unsafe
    /// Polls last index postiion. If used. use the address and the prepended reference to the 
    /// to return an TransferToken reference. Also sets the poll index to show the next item in list. 
    fn poll(&mut self) -> Option<Pinned<TransferToken>> {
        unimplemented!();
    }

    fn push_batch(&mut self, tkn_lst: Vec<TransferToken>) -> Vec<Pinned<TransferToken>> {
        todo!("implement batch push of ring");
    }

    fn push(&mut self, tkn: TransferToken) -> Pinned<TransferToken> {
        // fix memory address of token
        let mut pinned = Pinned::new(tkn);

        // Check length and if its fits. This should always be true due to the restriction of
        // the memory pool, but to be sure.
        assert!(pinned.buff_tkn.as_ref().unwrap().len() <= self.capacity);

        // create an counter that wrappes to the first element
        // after reaching a the end of the ring 
        let mut ctrl = self.get_write_ctrler();

        // write the descriptors in reversed order into the queue. Starting with recv descriptors.
        // As the device MUST see all readable descriptors, bevore any writable descriptors
        // See Virtio specification v1.1. - 2.7.17
        //
        // Importance here is:
        // * distinguish between Indirect and direct buffers
        // * write descriptors in the correct order
        // * make them available in the right order (reversed order or i.e. lastly where device polls)
        match (&pinned.buff_tkn.as_ref().unwrap().send_buff, &pinned.buff_tkn.as_ref().unwrap().recv_buff) {
            (Some(send_buff), Some(recv_buff)) => {
                // It is important to differentiate between indirect and direct descriptors here and if
                // send & recv descriptors are defined or only one of them. 
                match (send_buff.get_ctrl_desc(), recv_buff.get_ctrl_desc()) {
                    (Some(ctrl_desc), Some(_)) => {
                        // One indirect descriptor with only flag indirect set    
                        ctrl.write_desc(ctrl_desc, DescrFlags::VIRTQ_DESC_F_INDIRECT.into()); 
                    },
                    (None, None) => {
                        let mut buff_len = send_buff.as_slice().len() + recv_buff.as_slice().len();

                        for desc in send_buff.as_slice() {
                            if buff_len > 1 {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_NEXT.into());
                            } else {
                                ctrl.write_desc(desc, 0);
                            }
                            buff_len -= 1;
                        }

                        for desc in recv_buff.as_slice() {
                            if buff_len > 1 {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_NEXT & DescrFlags::VIRTQ_DESC_F_WRITE);
                            } else {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_WRITE.into());
                            }
                            buff_len -= 1;
                        } 
                    }
                    (None, Some(_)) => panic!("Indirect buffers mixed with direct buffers!"), // This should already be catched at creation of BufferToken
                    (Some(_), None) => panic!("Indirect buffers mixed with direct buffers!"), // This should already be catched at creation of BufferToken,
                }                
            },
            (Some(send_buff), None) => {
                match send_buff.get_ctrl_desc() {
                    Some(ctrl_desc) => {
                       // One indirect descriptor with only flag indirect set    
                       ctrl.write_desc(ctrl_desc, DescrFlags::VIRTQ_DESC_F_INDIRECT.into()); 
                    },
                    None => {
                        let mut buff_len = send_buff.as_slice().len();

                        for desc in send_buff.as_slice() {
                            if buff_len > 1 {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_NEXT.into());
                            } else {
                                ctrl.write_desc(desc, 0);
                            }
                            buff_len -= 1;
                        } 
                    }
                }
            },
            (None, Some(recv_buff)) => {
                match recv_buff.get_ctrl_desc() {
                    Some(ctrl_desc) => {
                       // One indirect descriptor with only flag indirect set    
                       ctrl.write_desc(ctrl_desc, DescrFlags::VIRTQ_DESC_F_INDIRECT.into()); 
                    },
                    None => {
                        let mut buff_len = recv_buff.as_slice().len();

                        for desc in recv_buff.as_slice() {
                            if buff_len > 1 {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_NEXT & DescrFlags::VIRTQ_DESC_F_WRITE);
                            } else {
                                ctrl.write_desc(desc, DescrFlags::VIRTQ_DESC_F_WRITE.into());
                            }
                            buff_len -= 1;
                        } 
                    }
                }
            },
            (None, None) => panic!("Empty Transfers are not allowed!"), // This should already be catched at creation of BufferToken
        }

        // Update flags of the first descriptor and set new write_index
        ctrl.make_avail(pinned.raw_addr());

        // Update the state of the actual Token
        pinned.state = TransferState::Processing;

        pinned
    }

    /// # Unsafe
    /// Returns the memory address of the first element of the descriptor ring
    fn raw_addr(&self) -> usize {
        self.ring.as_ptr() as usize
    }

    /// Returns an initalized write controler in order
    /// to write the queue correctly.
    fn get_write_ctrler(&mut self) -> WriteCtrl {
        WriteCtrl{
            start: self.write_index,
            position: self.write_index,
            modulo: self.ring.len(),
            wrap_at_init: self.wrap_count,
            buff_id: 0,

            desc_ring: self,
        }
    }
}


/// Convenient struct, allowing to increment and decrement inside the given 
/// modulo. Furthermore allows to convinently write descritpros into the queue.
/// 
/// **Example:**
/// 
/// The following code will count 6 steps backwards from 3 inside the modulo 9.
/// The output will be: 3,2,1,0,8,7.
///
/// ```
/// let mut writer = WriteCtrl {
///    val: 3,
///    modulo: 9
/// };
///
/// for i in 0..6 {
///    println!("{}", index.val);
///    index.decrmt();
/// }
/// ```
struct WriteCtrl<'a>{
    /// Where did the write of the buffer start in the descriptor ring
    /// This is important, as we must make this descriptor available 
    /// lastly.
    start: usize,
    /// Where to write next. This should always be equal to the Rings
    /// write_next field.
    position: usize,
    modulo: usize,
    /// What was the WrapCount at the first write position
    /// Important in order to set the right avail and used flags
    wrap_at_init: WrapCount,
    /// Buff ID of this write
    buff_id: u16,

    desc_ring: &'a mut DescriptorRing,
}


impl<'a> WriteCtrl<'a> {
    /// **This function MUST only be used within the WriteCtrl.write_desc() function!**
    ///
    /// Incrementing index by one. The index wrappes around to zero when 
    /// reaching (modulo -1).
    ///
    /// Also takes care of wrapping the WrapCount of the associated 
    /// DescriptorRing.
    fn incrmt(&mut self) {
        // Firstly check if we are at all allowed to write a descriptor
        assert!(self.desc_ring.capacity != 0);
        self.desc_ring.capacity -= 1;
        // check if increment wrapped around end of ring
        // then also wrap the wrap counter.
        if self.position + 1 == self.modulo {
            self.desc_ring.wrap_count.wrap();
        }
        // Also update the write_index
        self.desc_ring.write_index = (self.desc_ring.write_index + 1) % self.modulo;

        self.position = (self.position + 1) % self.modulo;
    }

    /// Writes a descriptor of a buffer into the queue. At the correct position, and 
    /// with the given flags.
    /// * Flags for avail and used will be set by the queue itself.
    ///   * -> Only set different flags here.
    fn write_desc(&mut self, mem_desc: &MemDescr, flags: u16) {
        // This also sets the buff_id for the WriteCtrl stuct to the ID of the first 
        // descriptor.
        if self.start == self.position {
            let desc_ref = &mut self.desc_ring.ring[self.position];
            desc_ref.address = mem_desc.ptr as u64;
            desc_ref.len = mem_desc.len as u32;
            desc_ref.buff_id = mem_desc.id.as_ref().unwrap().0; 
            // The driver performs a suitable memory barrier to ensure the device sees the updated descriptor table and available ring before the next step.
            // See Virtio specfification v1.1. - 2.7.21
            fence(Ordering::SeqCst);
            // Remove possibly set avail and used flags
            desc_ref.flags = flags & 0xFEFE;

            self.buff_id = mem_desc.id.as_ref().unwrap().0;
            self.incrmt();
        } else {
            let mut desc_ref = &mut self.desc_ring.ring[self.position];
            desc_ref.address = mem_desc.ptr as u64;
            desc_ref.len = mem_desc.len as u32;
            desc_ref.buff_id = self.buff_id;
            // The driver performs a suitable memory barrier to ensure the device sees the updated descriptor table and available ring before the next step.
            // See Virtio specfification v1.1. - 2.7.21
            fence(Ordering::SeqCst);
            // Remove possibly set avail and used flags and then set avail and used 
            // according to the current WrapCount.
            desc_ref.flags = (flags & 0xFEFE) | self.desc_ring.wrap_count.as_flags();

            self.incrmt()
        }
    }

    fn make_avail(&mut self, raw_tkn: *mut TransferToken) {
        // provide reference, in order to let TransferToken now upon finish.
        self.desc_ring.tkn_ref_ring[usize::try_from(self.buff_id).unwrap()] = raw_tkn;
        // The driver performs a suitable memory barrier to ensure the device sees the updated descriptor table and available ring before the next step.
        // See Virtio specfification v1.1. - 2.7.21
		fence(Ordering::SeqCst);
        self.desc_ring.ring[self.start].flags |= self.wrap_at_init.as_flags();
    }
}

#[repr(C, align(16))]
struct Descriptor {
    address: u64,
    len: u32,
    buff_id: u16,
    flags: u16,
}

impl Descriptor {
    fn new(add: u64, len: u32, id: u16, flags: u16) -> Self {
        Descriptor {
            address: add,
            len,
            buff_id: id,
            flags,
        }
    }

    fn to_le_bytes(self) -> [u8; 16] {
        let mut desc_bytes_cnt = 0usize;
        // 128 bits long raw descriptor bytes
        let mut desc_bytes: [u8; 16] = [0;16];

        // Call to little endian, as device will read this and
        // Virtio devices are inherently little endian coded.
        let mem_addr: [u8;8] = self.address.to_le_bytes();
        // Write address as bytes in raw
        for byte in 0..8 {
            desc_bytes[desc_bytes_cnt] = mem_addr[byte];
            desc_bytes_cnt += 1;
        }

        // Must be 32 bit in order to fulfill specification.
        // MemPool.pull and .pull_untracked ensure this automatically
        // which makes this cast safe.
        let mem_len: [u8; 4] = self.len.to_le_bytes();
        // Write length of memory area as bytes in raw
        for byte in 0..4 {
            desc_bytes[desc_bytes_cnt] = mem_len[byte];
            desc_bytes_cnt += 1;
        }

        // Write BuffID as bytes in raw.
        let id: [u8;2] = self.buff_id.to_le_bytes();
        for byte in 0..2usize {
            desc_bytes[desc_bytes_cnt] = id[byte];
            desc_bytes_cnt += 1;
        }

        // Write flags as bytes in raw.
        let flags: [u8; 2] = self.flags.to_le_bytes();
        // Write of flags as bytes in raw
        for byte in 0..2usize {
            desc_bytes[desc_bytes_cnt] = flags[byte];
        }

        desc_bytes
    }

    fn is_used() {
        unimplemented!();
    }
}

/// Driver and device event suppression struct used in packed virtqueues.
///
/// Structure layout see Virtio specification v1.1. - 2.7.14
/// Alignment see Virtio specification v1.1. - 2.7.10.1
#[repr(C, align(4))]
struct EventSuppr {
   event: u16,
   flags: u16, 
}

impl EventSuppr {
    /// Returns a zero initalized EventSuppr structure
    fn new() -> Self {
        EventSuppr {
            event: 0,
            flags: 0,
        }
    }
    
    /// Enables notifications by setting the LSB.
    /// See Virito specification v1.1. - 2.7.10
    fn enable_notif() {
        unimplemented!();
    }

    /// Disables notifications by unsetting the LSB.
    /// See Virtio specification v1.1. - 2.7.10
    fn disable_notif() {
        unimplemented!();
    }

    /// Reads notification bit (i.e. LSB) and returns value.
    /// If notifications are enabled returns true, else false.
    fn is_notif() -> bool {
        unimplemented!();
    }


    fn enable_specific(descriptor_id: u16, on_count: WrapCount) {
        // Check if VIRTIO_F_RING_EVENT_IDX has been negotiated

        // Check if descriptor_id is below 2^15

        // Set second bit from LSB to true

        // Set descriptor id, triggering notification

        // Set which wrap counter triggers

        unimplemented!();
    }
}

/// Packed virtqueue which provides the functionilaty as described in the 
/// virtio specification v1.1. - 2.7
pub struct PackedVq {
    /// Ring which allows easy access to the raw ring structure of the 
    /// specfification
    descr_ring: RefCell<DescriptorRing>,
    /// Raw EventSuppr structure
    drv_event: Box<EventSuppr>,
    /// Raw
    dev_event: Box<EventSuppr>,
    /// Memory pool controls the amount of "free floating" descriptors
    /// See [MemPool](super.MemPool) docs for detail.
    mem_pool: Rc<MemPool>,
    /// The size of the queue, equals the number of descriptors which can
    /// be used
    size: VqSize,
    /// The virtqueues index. This identifies the virtqueue to the 
    /// device and is unique on a per device basis
    index: VqIndex,
    /// Holds all erly dropped `TransferToken`
    /// If `TransferToken.state == TransferState::Finished`
    /// the Token can be safely dropped
    dropped: RefCell<Vec<Pinned<TransferToken>>>,
}



// Public interface of PackedVq
// This interface is also public in order to allow people to use the PackedVq directly!
// This is currently unlikely, as the Tokens hold a Rc<Virtq> for refering to their origin 
// queue. This could be eased 
impl PackedVq {
    pub fn early_drop(&self, tkn: Pinned<TransferToken>) {
        match tkn.state {
            TransferState::Finished => (), // Drop the pinned token -> Dealloc everything
            TransferState::Ready => panic!("Early dropped transfers are not allowed to be state == Ready"),
            TransferState::Processing => {
                // Keep token until state is finished. This needs to be checked/cleaned up later
                self.dropped.borrow_mut().push(tkn);
            }
        }
    }

    pub fn index(&self) -> VqIndex {
        self.index
    }

    pub fn new(com_cfg: &mut ComCfg, size: VqSize, index: VqIndex) -> Result<Self, VqPackedError> {
        // Get a handler to the queues configuration area.
        let mut vq_handler = match com_cfg.select_vq(index.into()) {
            Some(handler) => handler,
            None => return Err(VqPackedError::QueueNotExisting(index.into())),
        };

        // Must catch zero size as it is not allowed for packed queues.
        // Must catch size larger 32768 (2^15) as it is not allowed for packed queues.
        //
        // See Virtio specification v1.1. - 4.1.4.3.2
        let vq_size;
        if (size.0 == 0) | (size.0 > 32768) {
            return Err(VqPackedError::SizeNotAllowed(size.0));
        } else {
            vq_size = vq_handler.set_vq_size(size.0);
        }
        
        let descr_ring = RefCell::new(DescriptorRing::new(vq_size));
        let drv_event = Box::into_raw(Box::new(EventSuppr::new()));
        let dev_event= Box::into_raw(Box::new(EventSuppr::new()));Box::new(EventSuppr::new());

        // Provide memory areas of the queues data structures to the device
        vq_handler.set_ring_addr(index.into(), descr_ring.borrow().raw_addr());
        // As usize is safe here, as the *mut EventSuppr raw pointer is a thin pointer of size usize
        vq_handler.set_drv_ctrl_addr(index.into(), drv_event as usize);
        vq_handler.set_dev_ctrl_addr(index.into(), dev_event as usize);

        let drv_event = unsafe{Box::from_raw(drv_event)};
        let dev_event = unsafe{Box::from_raw(dev_event)};


        // Initalize new memory pool.
        let mem_pool = Rc::new(MemPool::new(size.0));

        // Initalize an empty vector for future dropped transfers
        let dropped: RefCell<Vec<Pinned<TransferToken>>> = RefCell::new(Vec::new());
    
        Ok(PackedVq {
            descr_ring,
            drv_event, 
            dev_event, 
            mem_pool,
            size,
            index,
            dropped,
        })
    }

    /// See `Virtq.prep_transfer()` documentation.
    pub fn dispatch(&self, tkn: TransferToken) -> Transfer {
        Transfer {
            transfer_tkn: Some(self.descr_ring.borrow_mut().push(tkn)),
        }
    }

    /// See `Virtq.prep_transfer()` documentation.
    pub fn prep_transfer<T: AsSliceU8 + 'static, K: AsSliceU8 + 'static>(&self, master: Rc<Virtq>, send: Option<(Box<T>, BuffSpec)>, recv: Option<(Box<K>, BuffSpec)>) 
        -> Result<TransferToken, VirtqError> {
        match (send, recv) {
            (None, None) => return Err(VirtqError::BufferNotSpecified),
            (Some((send_data, send_spec)), None) => {
                match send_spec {
                    BuffSpec::Single(size) => {
                        let data_slice = unsafe {send_data.as_slice_u8()};
                        let len = data_slice.len();

                        // Buffer must have the right size
                        if data_slice.len() != size.into() {
                            return Err(VirtqError::BufferSizeWrong(data_slice.len()))
                        }

                        let desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), data_slice,true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);

                        let buff_tkn = Some(BufferToken {
                            send_buff: Some(Buffer::Single{ desc_lst: vec![desc].into_boxed_slice(), len, next_write: 0 }),
                            recv_buff: None,
                            vq: master,
                            ret_send: true,
                            ret_recv: false,
                            reusable: true,
                        });

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn,
                            await_queue: None,
                        })
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let data_slice = unsafe {send_data.as_slice_u8()};
                        let len = data_slice.len();
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                       // Leak the box, as the memory will be deallocated upon drop of MemDescr
                       Box::leak(send_data); 

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: desc_lst.into_boxed_slice(), len, next_write: 0 }),
                                recv_buff: None,
                                vq: master,
                                ret_send: true,
                                ret_recv: false,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let data_slice = send_data.as_slice_u8();
                        let len = data_slice.len();
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, true));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let ctrl_desc = match self.create_indirect_ctrl(Some(&desc_lst), None) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);
                        
                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Indirect{ desc_lst: desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len, next_write: 0 }),
                                recv_buff: None,
                                vq: master,
                                ret_send: true,
                                ret_recv: false,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                }
            },
            (None, Some((recv_data, recv_spec))) => {
                match recv_spec {
                    BuffSpec::Single(size) => {
                        let data_slice = recv_data.as_slice_u8();
                        let len = data_slice.len();

                        // Buffer must have the right size
                        if data_slice.len() != size.into() {
                            return Err(VirtqError::BufferSizeWrong(data_slice.len()))
                        }

                        let desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), data_slice,true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![desc].into_boxed_slice(), len, next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let data_slice = unsafe {recv_data.as_slice_u8()};
                        let len = data_slice.len();
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                       // Leak the box, as the memory will be deallocated upon drop of MemDescr
                       Box::leak(recv_data); 

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Multiple{ desc_lst: desc_lst.into_boxed_slice(), len, next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let data_slice = unsafe {recv_data.as_slice_u8()};
                        let len = data_slice.len();
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, true));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let ctrl_desc = match self.create_indirect_ctrl(None, Some(&desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);
                        
                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Indirect{ desc_lst: desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len, next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                }
            },
            (Some((send_data, send_spec)), Some((recv_data, recv_spec))) => {
                match (send_spec, recv_spec) {
                    (BuffSpec::Single(send_size), BuffSpec::Single(recv_size)) => {
                        let send_data_slice = send_data.as_slice_u8();
                        let send_len = send_data_slice.len();

                        // Buffer must have the right size
                        if send_data_slice.len() != send_size.into() {
                            return Err(VirtqError::BufferSizeWrong(send_data_slice.len()))
                        }

                        let send_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), send_data_slice, true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);

                        let recv_data_slice = unsafe {recv_data.as_slice_u8()};
                        let recv_len = recv_data_slice.len();

                        // Buffer must have the right size
                        if recv_data_slice.len() != recv_size.into() {
                            return Err(VirtqError::BufferSizeWrong(recv_data_slice.len()))
                        }

                        let recv_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), recv_data_slice, true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_len, next_write: 0 }),
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_len, next_write: 0 }),
                                vq: master,
                                ret_send: true,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Single(send_size), BuffSpec::Multiple(recv_size_lst)) => {
                        let send_data_slice = unsafe {send_data.as_slice_u8()};
                        let send_len = send_data_slice.len();

                        // Buffer must have the right size
                        if send_data_slice.len() != send_size.into() {
                            return Err(VirtqError::BufferSizeWrong(send_data_slice.len()))
                        }

                        let send_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), send_data_slice, true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);

                        let recv_data_slice = unsafe {recv_data.as_slice_u8()};
                        let recv_len = recv_data_slice.len();
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                       // Leak the box, as the memory will be deallocated upon drop of MemDescr
                       Box::leak(recv_data);  

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_len, next_write: 0 }),
                                recv_buff: Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_len, next_write: 0 }),
                                vq: master,
                                ret_send: true,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Multiple(recv_size_lst)) => {
                        let send_data_slice = unsafe {send_data.as_slice_u8()};
                        let send_len = send_data_slice.len();
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);  

                        let recv_data_slice = unsafe {recv_data.as_slice_u8()};
                        let recv_len = recv_data_slice.len();
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);  

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_len, next_write: 0 }),
                                recv_buff: Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_len, next_write: 0 }),
                                vq: master,
                                ret_send: true,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Single(recv_size)) => {
                        let send_data_slice = unsafe {send_data.as_slice_u8()};
                        let send_len = send_data_slice.len();
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, true) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);  

                        let recv_data_slice = unsafe {recv_data.as_slice_u8()};
                        let recv_len = recv_data_slice.len();

                        // Buffer must have the right size
                        if recv_data_slice.len() != recv_size.into() {
                            return Err(VirtqError::BufferSizeWrong(recv_data_slice.len()))
                        }

                        let recv_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), recv_data_slice, true) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_len, next_write: 0 }),
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_len, next_write: 0 }),
                                vq: master,
                                ret_send: true,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Indirect(send_size_lst), BuffSpec::Indirect(recv_size_lst)) => {
                        let send_data_slice = unsafe {send_data.as_slice_u8()};
                        let send_len = send_data_slice.len();
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            send_desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, true));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(send_data);  

                        let recv_data_slice = unsafe {recv_data.as_slice_u8()};
                        let recv_len = recv_data_slice.len();
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            recv_desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, true));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        // Leak the box, as the memory will be deallocated upon drop of MemDescr
                        Box::leak(recv_data);  

                        let ctrl_desc = match self.create_indirect_ctrl(Some(&send_desc_lst), Some(&recv_desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        }; 

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                recv_buff: Some(Buffer::Indirect{ desc_lst: recv_desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc.no_dealloc_clone(), len: recv_len, next_write: 0 }),
                                send_buff: Some(Buffer::Indirect{ desc_lst: send_desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len: send_len, next_write: 0 }),
                                vq: master,
                                ret_send: true,
                                ret_recv: true,
                                reusable: true,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Indirect(_), BuffSpec::Single(_)) | (BuffSpec::Indirect(_), BuffSpec::Multiple(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    },
                    (BuffSpec::Single(_), BuffSpec::Indirect(_)) | (BuffSpec::Multiple(_), BuffSpec::Indirect(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    }
                }
            }
        }        
    }

    /// See `Virtq.prep_transfer_from_raw()` documentation.
    pub fn prep_transfer_from_raw<T: AsSliceU8 + 'static, K: AsSliceU8 + 'static>(&self, master: Rc<Virtq>, send: Option<(*mut T, BuffSpec)>, recv: Option<(*mut K, BuffSpec)>) 
        -> Result<TransferToken, VirtqError> {
        match (send, recv) {
            (None, None) => return Err(VirtqError::BufferNotSpecified),
            (Some((send_data, send_spec)), None) => {
                match send_spec {
                    BuffSpec::Single(size) => {
                        let data_slice = unsafe {(*send_data).as_slice_u8()};

                        // Buffer must have the right size
                        if data_slice.len() != size.into() {
                            return Err(VirtqError::BufferSizeWrong(data_slice.len()))
                        }

                        let desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Single{ desc_lst: vec![desc].into_boxed_slice(), len: data_slice.len(), next_write: 0 }),
                                recv_buff: None,
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let data_slice = unsafe {(*send_data).as_slice_u8()};
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        } 

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: desc_lst.into_boxed_slice(), len: data_slice.len(), next_write: 0 }),
                                recv_buff: None,
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let data_slice = unsafe {(*send_data).as_slice_u8()};
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, false));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let ctrl_desc = match self.create_indirect_ctrl(Some(&desc_lst), None) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Indirect{ desc_lst: desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len: data_slice.len(), next_write: 0 }),
                                recv_buff: None,
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                }
            },
            (None, Some((recv_data, recv_spec))) => {
                match recv_spec {
                    BuffSpec::Single(size) => {
                        let data_slice = unsafe {(*recv_data).as_slice_u8()};

                        // Buffer must have the right size
                        if data_slice.len() != size.into() {
                            return Err(VirtqError::BufferSizeWrong(data_slice.len()))
                        }

                        let desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![desc].into_boxed_slice(), len: data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let data_slice = unsafe {(*recv_data).as_slice_u8()};
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Multiple{ desc_lst: desc_lst.into_boxed_slice(), len: data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let data_slice = unsafe {(*recv_data).as_slice_u8()};
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut index = 0usize;

                        for byte in size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(data_slice.len())),
                            };

                            desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, false));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let ctrl_desc = match self.create_indirect_ctrl( None, Some(&desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };
                        
                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: None,
                                recv_buff: Some(Buffer::Indirect{ desc_lst: desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len: data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                }
            },
            (Some((send_data, send_spec)), Some((recv_data, recv_spec))) => {
                match (send_spec, recv_spec) {
                    (BuffSpec::Single(send_size), BuffSpec::Single(recv_size)) => {
                        let send_data_slice = unsafe {(*send_data).as_slice_u8()};

                        // Buffer must have the right size
                        if send_data_slice.len() != send_size.into() {
                            return Err(VirtqError::BufferSizeWrong(send_data_slice.len()))
                        }

                        let send_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), send_data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        let recv_data_slice = unsafe {(*recv_data).as_slice_u8()};

                        // Buffer must have the right size
                        if recv_data_slice.len() != recv_size.into() {
                            return Err(VirtqError::BufferSizeWrong(recv_data_slice.len()))
                        }

                        let recv_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), recv_data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_data_slice.len(), next_write: 0 }),
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Single(send_size), BuffSpec::Multiple(recv_size_lst)) => {
                        let send_data_slice = unsafe {(*send_data).as_slice_u8()};

                        // Buffer must have the right size
                        if send_data_slice.len() != send_size.into() {
                            return Err(VirtqError::BufferSizeWrong(send_data_slice.len()))
                        }

                        let send_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), send_data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        let recv_data_slice = unsafe {(*recv_data).as_slice_u8()};
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_data_slice.len(), next_write: 0 }),
                                recv_buff: Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Multiple(recv_size_lst)) => {
                        let send_data_slice = unsafe {(*send_data).as_slice_u8()};
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let recv_data_slice = unsafe {(*recv_data).as_slice_u8()};
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_data_slice.len(), next_write: 0 }),
                                recv_buff: Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Single(recv_size)) => {
                        let send_data_slice = unsafe {(*send_data).as_slice_u8()};
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), next_slice, false) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            };

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let recv_data_slice = unsafe {(*recv_data).as_slice_u8()};

                        // Buffer must have the right size
                        if recv_data_slice.len() != recv_size.into() {
                            return Err(VirtqError::BufferSizeWrong(recv_data_slice.len()))
                        }

                        let recv_desc = match self.mem_pool.pull_from(Rc::clone(&self.mem_pool), recv_data_slice, false) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                send_buff: Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_data_slice.len(), next_write: 0 }),
                                recv_buff: Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Indirect(send_size_lst), BuffSpec::Indirect(recv_size_lst)) => {
                        let send_data_slice = unsafe {(*send_data).as_slice_u8()};
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut index = 0usize;

                        for byte in send_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match send_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(send_data_slice.len())),
                            };

                            send_desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, false));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let recv_data_slice = unsafe {(*recv_data).as_slice_u8()};
                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut index = 0usize;

                        for byte in recv_size_lst {
                            let end_index = index + usize::from(*byte);
                            let next_slice = match recv_data_slice.get(index..end_index){
                                Some(slice) => slice, 
                                None => return Err(VirtqError::BufferSizeWrong(recv_data_slice.len())),
                            };

                            recv_desc_lst.push(self.mem_pool.pull_from_untracked(Rc::clone(&self.mem_pool), next_slice, false));

                            // update the starting index for the next iteration
                            index = index + usize::from(*byte);
                        }

                        let ctrl_desc = match self.create_indirect_ctrl( Some(&send_desc_lst), Some(&recv_desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        }; 

                        Ok(TransferToken{
                            state: TransferState::Ready,
                            buff_tkn: Some(BufferToken {
                                recv_buff: Some(Buffer::Indirect{ desc_lst: recv_desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc.no_dealloc_clone(), len: recv_data_slice.len(), next_write: 0 }),
                                send_buff: Some(Buffer::Indirect{ desc_lst: send_desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc, len: send_data_slice.len(), next_write: 0 }),
                                vq: master,
                                ret_send: false,
                                ret_recv: false,
                                reusable: false,
                            }),
                            await_queue: None,
                        })
                    },
                    (BuffSpec::Indirect(_), BuffSpec::Single(_)) | (BuffSpec::Indirect(_), BuffSpec::Multiple(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    },
                    (BuffSpec::Single(_), BuffSpec::Indirect(_)) | (BuffSpec::Multiple(_), BuffSpec::Indirect(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    }
                }
            }
        } 
    }

    /// See `Virtq.prep_buffer()` documentation.
    pub fn prep_buffer(&self, master: Rc<Virtq>, send: Option<BuffSpec>, recv: Option<BuffSpec>) 
        -> Result<BufferToken, VirtqError> {
        match (send, recv) {
            // No buffers specified
            (None, None) => return Err(VirtqError::BufferNotSpecified),
            // Send buffer specified, No recv buffer
            (Some(spec), None) => {
                match spec {
                    BuffSpec::Single(size) => match self.mem_pool.pull(Rc::clone(&self.mem_pool), size) {
                        Ok(desc) => {
                            let buffer = Buffer::Single{desc_lst: vec![desc].into_boxed_slice(), len: size.into(), next_write: 0};

                            Ok(BufferToken {
                                send_buff: Some(buffer),
                                recv_buff: None,
                                vq: master,
                                ret_send: true,
                                ret_recv: false,
                                reusable: true,
                            })
                        }
                        Err(vq_err) => return Err(vq_err),
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut len = 0usize;

                        for size in size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            len += usize::from(*size);
                        }

                        let buffer = Buffer::Multiple{desc_lst: desc_lst.into_boxed_slice(), len, next_write: 0};

                        Ok(BufferToken{
                            send_buff: Some(buffer),
                            recv_buff: None,
                            vq: master,
                            ret_send: true,
                            ret_recv: false,
                            reusable: true,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut len = 0usize;

                        for size in size_lst {
                            // As the indirect list does only consume one descriptor for the 
                            // control descriptor, the actual list is untracked
                            desc_lst.push(
                                self.mem_pool.pull_untracked(Rc::clone(&self.mem_pool), *size)
                            );
                            len += usize::from(*size);;
                        }

                        let ctrl_desc = match self.create_indirect_ctrl( Some(&desc_lst), None) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };
                        
                        let buffer = Buffer::Indirect{desc_lst: desc_lst.into_boxed_slice(), ctrl_desc, len, next_write: 0};

                        Ok(BufferToken{
                            send_buff: Some(buffer),
                            recv_buff: None,
                            vq: master,
                            ret_send: true,
                            ret_recv: false,
                            reusable: true,
                        }) 
                    },
                }
            },
            // No send buffer, recv buffer is specified
            (None, Some(spec)) => {
                match spec {
                    BuffSpec::Single(size) => match self.mem_pool.pull(Rc::clone(&self.mem_pool), size) {
                        Ok(desc) => {
                            let buffer = Buffer::Single{desc_lst: vec![desc].into_boxed_slice(), len: size.into(), next_write: 0};

                            Ok(BufferToken {
                                send_buff: None,
                                recv_buff: Some(buffer),
                                vq: master,
                                ret_send: false,
                                ret_recv: true,
                                reusable: true,
                            })
                        }
                        Err(vq_err) => return Err(vq_err),
                    },
                    BuffSpec::Multiple(size_lst) => {
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut len = 0usize;

                        for size in size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            len += usize::from(*size);;
                        }

                        let buffer = Buffer::Multiple{desc_lst: desc_lst.into_boxed_slice(), len, next_write: 0};

                        Ok(BufferToken{
                            send_buff: None,
                            recv_buff: Some(buffer),
                            vq: master,
                            ret_send: false,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                    BuffSpec::Indirect(size_lst) => {
                        let mut desc_lst: Vec<MemDescr> = Vec::with_capacity(size_lst.len());
                        let mut len = 0usize;

                        for size in size_lst {
                            // As the indirect list does only consume one descriptor for the 
                            // control descriptor, the actual list is untracked
                            desc_lst.push(
                                self.mem_pool.pull_untracked(Rc::clone(&self.mem_pool), *size)
                            );
                            len += usize::from(*size);
                        }

                        let ctrl_desc =  match self.create_indirect_ctrl( None, Some(&desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };
                        
                        let buffer = Buffer::Indirect{desc_lst: desc_lst.into_boxed_slice(), ctrl_desc, len, next_write: 0};

                        Ok(BufferToken{
                            send_buff: None,
                            recv_buff: Some(buffer),
                            vq: master,
                            ret_send: false,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                }
            },
            // Send buffer specified, recv buffer specified
            (Some(send_spec), Some(recv_spec)) => {
                match (send_spec, recv_spec) {
                    (BuffSpec::Single(send_size), BuffSpec::Single(recv_size)) => {
                        let send_buff = match self.mem_pool.pull(Rc::clone(&self.mem_pool), send_size) {
                            Ok(send_desc) => {
                                Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_size.into(), next_write: 0 })
                            }
                            Err(vq_err) => return Err(vq_err),
                        };

                        let recv_buff = match self.mem_pool.pull(Rc::clone(&self.mem_pool), recv_size) {
                            Ok(recv_desc) => {
                                Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_size.into(), next_write: 0 })
                            }
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(BufferToken{
                            send_buff,
                            recv_buff,
                            vq: master,
                            ret_send: true,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                    (BuffSpec::Single(send_size), BuffSpec::Multiple(recv_size_lst)) => {
                        let send_buff = match self.mem_pool.pull(Rc::clone(&self.mem_pool), send_size) {
                            Ok(send_desc) => {
                                Some(Buffer::Single{ desc_lst: vec![send_desc].into_boxed_slice(), len: send_size.into(), next_write: 0 })
                            }
                            Err(vq_err) => return Err(vq_err),
                        };

                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut recv_len = 0usize;

                        for size in recv_size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            recv_len += usize::from(*size);
                        }

                        let recv_buff = Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_len , next_write: 0 });

                        Ok(BufferToken{
                            send_buff,
                            recv_buff,
                            vq: master,
                            ret_send: true,
                            ret_recv: true,
                            reusable: true,
                        })

                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Multiple(recv_size_lst)) => {
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut send_len = 0usize;
                        for size in send_size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            send_len += usize::from(*size);
                        }

                        let send_buff = Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_len , next_write: 0 });

                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut recv_len = 0usize;

                        for size in recv_size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => recv_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            recv_len += usize::from(*size);
                        }

                        let recv_buff = Some(Buffer::Multiple{ desc_lst: recv_desc_lst.into_boxed_slice(), len: recv_len , next_write: 0 });

                        Ok(BufferToken{
                            send_buff,
                            recv_buff,
                            vq: master,
                            ret_send: true,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                    (BuffSpec::Multiple(send_size_lst), BuffSpec::Single(recv_size)) => {
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut send_len = 0usize;

                        for size in send_size_lst {
                            match self.mem_pool.pull(Rc::clone(&self.mem_pool), *size) {
                                Ok(desc) => send_desc_lst.push(desc),
                                Err(vq_err) => return Err(vq_err),
                            }
                            send_len += usize::from(*size);
                        }

                        let send_buff = Some(Buffer::Multiple{ desc_lst: send_desc_lst.into_boxed_slice(), len: send_len , next_write: 0 });

                        let recv_buff = match self.mem_pool.pull(Rc::clone(&self.mem_pool), recv_size) {
                            Ok(recv_desc) => {
                                Some(Buffer::Single{ desc_lst: vec![recv_desc].into_boxed_slice(), len: recv_size.into(), next_write: 0 })
                            }
                            Err(vq_err) => return Err(vq_err),
                        };

                        Ok(BufferToken{
                            send_buff,
                            recv_buff,
                            vq: master,
                            ret_send: true,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                    (BuffSpec::Indirect(send_size_lst), BuffSpec::Indirect(recv_size_lst)) => {
                        let mut send_desc_lst: Vec<MemDescr> = Vec::with_capacity(send_size_lst.len());
                        let mut send_len = 0usize;

                        for size in send_size_lst {
                            // As the indirect list does only consume one descriptor for the 
                            // control descriptor, the actual list is untracked
                            send_desc_lst.push(
                                self.mem_pool.pull_untracked(Rc::clone(&self.mem_pool), *size)
                            );
                            send_len += usize::from(*size);
                        }

                        let mut recv_desc_lst: Vec<MemDescr> = Vec::with_capacity(recv_size_lst.len());
                        let mut recv_len = 0usize;

                        for size in recv_size_lst {
                            // As the indirect list does only consume one descriptor for the 
                            // control descriptor, the actual list is untracked
                            recv_desc_lst.push(
                                self.mem_pool.pull_untracked(Rc::clone(&self.mem_pool), *size)
                            );
                            recv_len += usize::from(*size);
                        }

                        let ctrl_desc =  match self.create_indirect_ctrl( Some(&send_desc_lst), Some(&recv_desc_lst)) {
                            Ok(desc) => desc,
                            Err(vq_err) => return Err(vq_err),
                        };
                        
                        let recv_buff = Some(Buffer::Indirect{ desc_lst: recv_desc_lst.into_boxed_slice(), ctrl_desc: ctrl_desc.no_dealloc_clone(), len: recv_len , next_write: 0 });
                        let send_buff = Some(Buffer::Indirect{ desc_lst: send_desc_lst.into_boxed_slice(), ctrl_desc, len: send_len , next_write: 0 });

                        Ok(BufferToken{
                            send_buff,
                            recv_buff,
                            vq: master,
                            ret_send: true,
                            ret_recv: true,
                            reusable: true,
                        })
                    },
                    (BuffSpec::Indirect(_), BuffSpec::Single(_)) | (BuffSpec::Indirect(_), BuffSpec::Multiple(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    },
                    (BuffSpec::Single(_), BuffSpec::Indirect(_)) | (BuffSpec::Multiple(_), BuffSpec::Indirect(_)) => {
                        return Err(VirtqError::BufferInWithDirect)
                    }
                }
            },
        }
    }

    pub fn size(&self) -> VqSize {
        self.size
    }
}

// Private Interface for PackedVq
impl PackedVq {
    fn create_indirect_ctrl(&self, send: Option<&Vec<MemDescr>>, recv: Option<&Vec<MemDescr>>) -> Result<MemDescr, VirtqError>{
        // Need to match (send, recv) twice, as the "size" of the control descriptor to be pulled must be known in advance.
        let len: usize;
        match (send, recv) {
            (None, None) => return Err(VirtqError::BufferNotSpecified),
            (None, Some(recv_desc_lst)) => {
                len = recv_desc_lst.len();
            },
            (Some(send_desc_lst), None) => {
                len = send_desc_lst.len();
            },
            (Some(send_desc_lst), Some(recv_desc_lst)) => {
                len = send_desc_lst.len() + recv_desc_lst.len();
            },
        }

        let sz_indrct_lst = Bytes(core::mem::size_of::<Descriptor>() * len);
        let mut ctrl_desc = match self.mem_pool.pull(Rc::clone(&self.mem_pool), sz_indrct_lst) {
            Ok(desc) => desc,
            Err(vq_err) => return Err(vq_err),
        };

        // For indexing into the allocated memory area. This reduces the 
        // function to only iterate over the MemDescr once and not twice
        // as otherwise needed if the raw descriptor bytes were to be stored
        // in an array.
        let mut crtl_desc_iter = 0usize;

        match (send, recv) {
            (None, None) => return Err(VirtqError::BufferNotSpecified),
            // Only recving descriptorsn (those are writabel by device)
            (None, Some(recv_desc_lst)) => {
                for desc in recv_desc_lst {
                   let raw: [u8; 16] = Descriptor::new(
                        (desc.ptr as u64),
                        (desc.len as u32),
                        0,
                        DescrFlags::VIRTQ_DESC_F_WRITE.into()
                   ).to_le_bytes();
                   
                   for byte in 0..16 {
                       ctrl_desc[crtl_desc_iter] = raw[byte];
                       crtl_desc_iter += 1;
                   }
                }
                Ok(ctrl_desc)
            },
            // Only sending descritpors
            (Some(send_desc_lst), None) => {
                for desc in send_desc_lst {
                    let raw: [u8; 16] = Descriptor::new(
                        (desc.ptr as u64),
                        (desc.len as u32),
                        0,
                        0, 
                   ).to_le_bytes();
                   
                   for byte in 0..16 {
                       ctrl_desc[crtl_desc_iter] = raw[byte];
                       crtl_desc_iter += 1;
                   }
                }
                Ok(ctrl_desc)
            },
            (Some(send_desc_lst), Some(recv_desc_lst)) => {
                // Send descriptors ALWAYS before receiving ones.
                for desc in send_desc_lst {
                    let raw: [u8; 16] = Descriptor::new(
                        (desc.ptr as u64),
                        (desc.len as u32),
                        0,
                        0, 
                   ).to_le_bytes();
                   
                   for byte in 0..16 {
                       ctrl_desc[crtl_desc_iter] = raw[byte];
                       crtl_desc_iter += 1;
                   }
                }

                for desc in recv_desc_lst {
                    let raw: [u8; 16] = Descriptor::new(
                        (desc.ptr as u64),
                        (desc.len as u32),
                        0,
                        DescrFlags::VIRTQ_DESC_F_WRITE.into()
                   ).to_le_bytes();
                   
                   for byte in 0..16 {
                       ctrl_desc[crtl_desc_iter] = raw[byte];
                       crtl_desc_iter += 1;
                   }
                }

                Ok(ctrl_desc)
            },
        }
    }
}

impl Drop for PackedVq {
    fn drop(&mut self) {
        todo!("rerutn leaked memory and ensure deallocation")
    }
}

pub mod error {
    pub enum VqPackedError {
        General,
        SizeNotAllowed(u16),
        QueueNotExisting(u16)
    }
}