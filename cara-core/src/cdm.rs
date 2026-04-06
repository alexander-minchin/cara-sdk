use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use regex::Regex;
use nalgebra::{Vector3, Matrix6};
use crate::utils::time_transformations::timestring_to_jd;

#[derive(Debug, Default, Clone)]
pub struct CdmHeader {
    pub version: String,
    pub creation_date: String,
    pub originator: String,
    pub message_id: String,
    pub tca: String,
    pub tca_jd: f64,
    pub miss_distance: f64,
    pub relative_speed: f64,
    pub relative_position: Vector3<f64>,
    pub relative_velocity: Vector3<f64>,
    pub collision_probability: f64,
    pub collision_probability_method: String,
    pub hbr: Option<f64>,
}

#[derive(Debug, Default, Clone)]
pub struct CdmObject {
    pub object: String,
    pub object_designator: String,
    pub catalog_name: String,
    pub object_name: String,
    pub international_designator: String,
    pub object_type: String,
    pub ref_frame: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub x_dot: f64,
    pub y_dot: f64,
    pub z_dot: f64,
    pub covariance: Matrix6<f64>,
}

#[derive(Debug, Clone)]
pub struct Cdm {
    pub header: CdmHeader,
    pub object1: CdmObject,
    pub object2: CdmObject,
}

pub fn parse_cdm_kvn<P: AsRef<Path>>(path: P) -> Result<Cdm, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    
    // Regex to capture key and value, ignoring potential unit brackets
    let re_kv = Regex::new(r"^\s*([^=]+)\s*=\s*([^\[\n\r]+)").unwrap();

    let mut header = CdmHeader::default();
    let mut objects: Vec<CdmObject> = Vec::new();
    let mut current_object = CdmObject::default();
    let mut in_object = false;

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(caps) = re_kv.captures(line) {
            let key = caps.get(1).unwrap().as_str().trim();
            let value = caps.get(2).unwrap().as_str().trim();

            match key {
                "CCSDS_CDM_VERS" => header.version = value.to_string(),
                "CREATION_DATE" => header.creation_date = value.to_string(),
                "ORIGINATOR" => header.originator = value.to_string(),
                "MESSAGE_ID" => header.message_id = value.to_string(),
                "TCA" => {
                    header.tca = value.to_string();
                    header.tca_jd = timestring_to_jd(value);
                }
                "MISS_DISTANCE" => header.miss_distance = value.parse().unwrap_or(0.0),
                "RELATIVE_SPEED" => header.relative_speed = value.parse().unwrap_or(0.0),
                "RELATIVE_POSITION_R" => header.relative_position[0] = value.parse().unwrap_or(0.0),
                "RELATIVE_POSITION_T" => header.relative_position[1] = value.parse().unwrap_or(0.0),
                "RELATIVE_POSITION_N" => header.relative_position[2] = value.parse().unwrap_or(0.0),
                "RELATIVE_VELOCITY_R" => header.relative_velocity[0] = value.parse().unwrap_or(0.0),
                "RELATIVE_VELOCITY_T" => header.relative_velocity[1] = value.parse().unwrap_or(0.0),
                "RELATIVE_VELOCITY_N" => header.relative_velocity[2] = value.parse().unwrap_or(0.0),
                "COLLISION_PROBABILITY" => header.collision_probability = value.parse().unwrap_or(0.0),
                "COLLISION_PROBABILITY_METHOD" => header.collision_probability_method = value.to_string(),
                "COMMENT HBR" => header.hbr = Some(value.parse().unwrap_or(0.0)),
                
                "OBJECT" => {
                    if in_object {
                        objects.push(current_object);
                    }
                    current_object = CdmObject::default();
                    current_object.object = value.to_string();
                    in_object = true;
                }
                "OBJECT_DESIGNATOR" => current_object.object_designator = value.to_string(),
                "CATALOG_NAME" => current_object.catalog_name = value.to_string(),
                "OBJECT_NAME" => current_object.object_name = value.to_string(),
                "INTERNATIONAL_DESIGNATOR" => current_object.international_designator = value.to_string(),
                "OBJECT_TYPE" => current_object.object_type = value.to_string(),
                "REF_FRAME" => current_object.ref_frame = value.to_string(),
                "X" => current_object.x = value.parse().unwrap_or(0.0),
                "Y" => current_object.y = value.parse().unwrap_or(0.0),
                "Z" => current_object.z = value.parse().unwrap_or(0.0),
                "X_DOT" => current_object.x_dot = value.parse().unwrap_or(0.0),
                "Y_DOT" => current_object.y_dot = value.parse().unwrap_or(0.0),
                "Z_DOT" => current_object.z_dot = value.parse().unwrap_or(0.0),
                
                "CR_R" => current_object.covariance[(0, 0)] = value.parse().unwrap_or(0.0),
                "CT_R" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(1, 0)] = v;
                    current_object.covariance[(0, 1)] = v;
                }
                "CT_T" => current_object.covariance[(1, 1)] = value.parse().unwrap_or(0.0),
                "CN_R" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(2, 0)] = v;
                    current_object.covariance[(0, 2)] = v;
                }
                "CN_T" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(2, 1)] = v;
                    current_object.covariance[(1, 2)] = v;
                }
                "CN_N" => current_object.covariance[(2, 2)] = value.parse().unwrap_or(0.0),
                
                "CRDOT_R" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(3, 0)] = v;
                    current_object.covariance[(0, 3)] = v;
                }
                "CTDOT_R" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(4, 0)] = v;
                    current_object.covariance[(0, 4)] = v;
                }
                "CNDOT_R" => {
                    let v = value.parse().unwrap_or(0.0);
                    current_object.covariance[(5, 0)] = v;
                    current_object.covariance[(0, 5)] = v;
                }
                "CRDOT_RDOT" => current_object.covariance[(3, 3)] = value.parse().unwrap_or(0.0),
                "CTDOT_TDOT" => current_object.covariance[(4, 4)] = value.parse().unwrap_or(0.0),
                "CNDOT_NDOT" => current_object.covariance[(5, 5)] = value.parse().unwrap_or(0.0),
                
                _ => {}
            }
        }
    }

    if in_object {
        objects.push(current_object);
    }

    if objects.len() < 2 {
        return Err("Expected at least 2 objects in CDM".to_string());
    }

    Ok(Cdm {
        header,
        object1: objects[0].clone(),
        object2: objects[1].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_cdm() {
        let path = "../upstream/DataFiles/SampleCDMs/AlfanoTestCase01.cdm";
        let cdm = parse_cdm_kvn(path).unwrap();
        
        assert_eq!(cdm.header.version, "1.0");
        assert_eq!(cdm.header.originator, "JSPOC");
        assert_eq!(cdm.header.hbr, Some(15.0));
        assert_eq!(cdm.object1.object_designator, "1001");
        assert_eq!(cdm.object2.object_designator, "1002");
        assert_relative_eq!(cdm.object1.x, 153.446765);
        assert_relative_eq!(cdm.object1.covariance[(0,0)], 19.88970273925819);
    }
}
