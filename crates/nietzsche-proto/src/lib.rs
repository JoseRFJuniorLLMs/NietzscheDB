#![allow(clippy::pedantic)]
#![allow(clippy::all)]
#![allow(unexpected_cfgs)]

pub mod db {
    tonic::include_proto!("db");
}
