// Copyright (c) 2018 Colin Finck, RWTH Aachen University
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use arch;
use errno::*;
use syscalls::sys_usleep;


#[repr(C)]
pub struct itimerval {
	pub it_interval: timeval,
	pub it_value: timeval,
}

#[repr(C)]
pub struct timespec {
	pub tv_sec: i64,
	pub tv_nsec: i64,
}

#[repr(C)]
pub struct timeval {
	pub tv_sec: i64,
	pub tv_usec: i64,
}

const CLOCK_REALTIME: u64 = 1;
const CLOCK_PROCESS_CPUTIME_ID: u64 = 2;
const CLOCK_THREAD_CPUTIME_ID: u64 = 3;
const CLOCK_MONOTONIC: u64 = 4;
const TIMER_ABSTIME: i32 = 4;


#[no_mangle]
pub extern "C" fn sys_clock_getres(clock_id: u64, res: *mut timespec) -> i32 {
	assert!(!res.is_null(), "sys_clock_getres called with a zero res parameter");
	let result = unsafe { &mut *res };

	match clock_id {
		CLOCK_REALTIME | CLOCK_PROCESS_CPUTIME_ID | CLOCK_THREAD_CPUTIME_ID | CLOCK_MONOTONIC => {
			// All clocks in HermitCore have 1 microsecond resolution.
			result.tv_sec = 0;
			result.tv_nsec = 1000;
			0
		},
		_ => {
			debug!("Called sys_clock_getres for unsupported clock {}", clock_id);
			-EINVAL
		}
	}
}

#[no_mangle]
pub extern "C" fn sys_clock_gettime(clock_id: u64, tp: *mut timespec) -> i32 {
	assert!(!tp.is_null(), "sys_clock_gettime called with a zero tp parameter");
	let result = unsafe { &mut *tp };

	match clock_id {
		CLOCK_MONOTONIC => {
			let microseconds = arch::processor::get_timer_ticks();
			result.tv_sec = (microseconds / 1_000_000) as i64;
			result.tv_nsec = ((microseconds % 1_000_000) * 1000) as i64;
			0
		},
		_ => {
			debug!("Called sys_clock_gettime for unsupported clock {}", clock_id);
			-EINVAL
		}
	}
}

#[no_mangle]
pub extern "C" fn sys_clock_nanosleep(clock_id: u64, flags: i32, rqtp: *const timespec, _rmtp: *mut timespec) -> i32 {
	assert!(!rqtp.is_null(), "sys_clock_nanosleep called with a zero rqtp parameter");
	let requested_time = unsafe { & *rqtp };
	if requested_time.tv_sec < 0 || requested_time.tv_nsec < 0 || requested_time.tv_nsec > 999_999_999 {
		debug!("sys_clock_nanosleep called with an invalid requested time, returning -EINVAL");
		return -EINVAL;
	}

	match clock_id {
		CLOCK_REALTIME | CLOCK_MONOTONIC => {
			let mut microseconds = (requested_time.tv_sec as u64) * 1_000_000 + (requested_time.tv_nsec as u64) / 1_000;

			if flags & TIMER_ABSTIME > 0 {
				if clock_id == CLOCK_MONOTONIC {
					microseconds -= arch::processor::get_timer_ticks();
				} else {
					// HermitCore does not yet know about the time since the Unix epoch.
					debug!("TIMER_ABSTIME for CLOCK_REALTIME is unimplemented, returning -EINVAL");
					return -EINVAL;
				}
			}

			sys_usleep(microseconds);
			0
		},
		_ => {
			-EINVAL
		}
	}
}

#[no_mangle]
pub extern "C" fn sys_clock_settime(_clock_id: u64, _tp: *const timespec) -> i32 {
	// We don't support setting any clocks yet.
	debug!("sys_clock_settime is unimplemented, returning -EINVAL");
	-EINVAL
}

#[no_mangle]
pub extern "C" fn sys_gettimeofday(tp: *mut timeval, tz: usize) -> i32 {
	if let Some(result) = unsafe { tp.as_mut() } {
		// We don't know the real time yet, so return a monotonic clock time starting at boot-up.
		let microseconds = arch::processor::get_timer_ticks();
		result.tv_sec = (microseconds / 1_000_000) as i64;
		result.tv_usec = (microseconds % 1_000_000) as i64;
	}

	if tz > 0 {
		debug!("The tz parameter in sys_gettimeofday is unimplemented, returning -EINVAL");
		return -EINVAL;
	}

	0
}

#[no_mangle]
pub extern "C" fn sys_setitimer(_which: i32, _value: *const itimerval, _ovalue: *mut itimerval) -> i32 {
	debug!("Called sys_setitimer, which is unimplemented and always returns 0");
	0
}
