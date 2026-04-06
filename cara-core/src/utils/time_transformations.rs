use chrono::{DateTime, Utc, NaiveDateTime};

/// Convert a UTC datetime to a Julian Date.
pub fn datetime_to_jd(dt: DateTime<Utc>) -> f64 {
    let year = dt.format("%Y").to_string().parse::<i32>().unwrap();
    let month = dt.format("%m").to_string().parse::<u32>().unwrap();
    let day = dt.format("%d").to_string().parse::<u32>().unwrap();
    let hour = dt.format("%H").to_string().parse::<u32>().unwrap();
    let minute = dt.format("%M").to_string().parse::<u32>().unwrap();
    let second = dt.format("%S").to_string().parse::<u32>().unwrap();
    let nanosecond = dt.format("%f").to_string().parse::<u32>().unwrap();

    let a = (14 - month as i32) / 12;
    let y = year + 4800 - a;
    let m = month as i32 + 12 * a - 3;

    let jd_int = day as i32 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    
    let jd_fraction = (hour as f64 - 12.0) / 24.0 
        + minute as f64 / 1440.0 
        + (second as f64 + nanosecond as f64 / 1_000_000_000.0) / 86400.0;
    
    jd_int as f64 + jd_fraction
}

/// Convert a time string (YYYY-MM-DD HH:MM:SS) to a Julian Date.
///
/// Ported from MATLAB: timestring_to_jd.m
pub fn timestring_to_jd(timestring: &str) -> f64 {
    // Normalize format: Replace T with space
    let normalized = timestring.replace('T', " ");
    
    // Try standard formats
    let dt = if let Ok(dt) = NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M:%S%.f") {
        DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M:%S") {
        DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(&normalized, "%Y-%j %H:%M:%S%.f") {
        DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(&normalized, "%Y-%j %H:%M:%S") {
        DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
    } else {
        panic!("Failed to parse timestring: {}", timestring);
    };

    datetime_to_jd(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    #[test]
    fn test_datetime_to_jd() {
        // 2000-01-01 12:00:00 UTC is exactly JD 2451545.0
        let dt = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap();
        assert_relative_eq!(datetime_to_jd(dt), 2451545.0);
    }

    #[test]
    fn test_timestring_to_jd() {
        assert_relative_eq!(timestring_to_jd("2000-01-01 12:00:00"), 2451545.0);
        assert_relative_eq!(timestring_to_jd("2000-01-01T12:00:00"), 2451545.0);
    }
}
