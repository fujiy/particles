#[cfg(unix)]
pub fn process_cpu_time_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a valid rusage into the provided pointer on success.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: rc == 0 guarantees the structure has been initialized by getrusage.
    let usage = unsafe { usage.assume_init() };
    let user = usage.ru_utime;
    let sys = usage.ru_stime;
    Some(
        user.tv_sec as f64
            + user.tv_usec as f64 * 1e-6
            + sys.tv_sec as f64
            + sys.tv_usec as f64 * 1e-6,
    )
}

#[cfg(not(unix))]
pub fn process_cpu_time_seconds() -> Option<f64> {
    None
}
