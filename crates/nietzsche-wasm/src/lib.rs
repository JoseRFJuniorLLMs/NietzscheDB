use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use nietzsche_core::{CosineMetric, EuclideanMetric, LorentzMetric, PoincareMetric, GlobalConfig, QuantizationMode};
use nietzsche_hnsw::HnswIndex;
use nietzsche_vecstore::VectorStore;
use rexie::{ObjectStore, Rexie, TransactionMode};

enum IndexWrapper {
    // Euclidean (L2)
    L2Dim384(Arc<HnswIndex<384, EuclideanMetric>>),
    L2Dim768(Arc<HnswIndex<768, EuclideanMetric>>),
    L2Dim1024(Arc<HnswIndex<1024, EuclideanMetric>>),
    L2Dim1536(Arc<HnswIndex<1536, EuclideanMetric>>),
    L2Dim3072(Arc<HnswIndex<3072, EuclideanMetric>>),
    // Cosine
    CosineDim384(Arc<HnswIndex<384, CosineMetric>>),
    CosineDim768(Arc<HnswIndex<768, CosineMetric>>),
    CosineDim1024(Arc<HnswIndex<1024, CosineMetric>>),
    CosineDim1536(Arc<HnswIndex<1536, CosineMetric>>),
    CosineDim3072(Arc<HnswIndex<3072, CosineMetric>>),
    // Poincare (Hyperbolic)
    PoincareDim384(Arc<HnswIndex<384, PoincareMetric>>),
    PoincareDim768(Arc<HnswIndex<768, PoincareMetric>>),
    PoincareDim1024(Arc<HnswIndex<1024, PoincareMetric>>),
    PoincareDim1536(Arc<HnswIndex<1536, PoincareMetric>>),
    PoincareDim3072(Arc<HnswIndex<3072, PoincareMetric>>),
    // Lorentz (Hyperboloid)
    LorentzDim384(Arc<HnswIndex<384, LorentzMetric>>),
    LorentzDim768(Arc<HnswIndex<768, LorentzMetric>>),
    LorentzDim1024(Arc<HnswIndex<1024, LorentzMetric>>),
    LorentzDim1536(Arc<HnswIndex<1536, LorentzMetric>>),
    LorentzDim3072(Arc<HnswIndex<3072, LorentzMetric>>),
}

const DB_NAME: &str = "nietzsche_db";
const STORE_NAME: &str = "storage"; // Object Store name

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct NietzscheDB {
    index: IndexWrapper,
    // Mapping UserID -> InternalID
    id_map: RwLock<HashMap<u32, u32>>,
    // Reverse mapping InternalID -> UserID
    rev_map: RwLock<HashMap<u32, u32>>,
    dimension: usize,
}

#[wasm_bindgen]
impl NietzscheDB {
    /// Creates a new `NietzscheDB` instance.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, metric: String) -> Result<NietzscheDB, JsValue> {
        console_error_panic_hook::set_once();

        // Use RAM implementation
        // Element size depends on dimension (Scalar f32 = 4 bytes)
        let element_size = dimension * 4;
        let storage = Arc::new(VectorStore::new(std::path::Path::new("mem"), element_size));
        let config = Arc::new(GlobalConfig::default());
        let mode = QuantizationMode::None;
        let metric = metric.to_lowercase();

        let index = match (dimension, metric.as_str()) {
             // Euclidean (L2)
             (384, "l2" | "euclidean") => IndexWrapper::L2Dim384(Arc::new(HnswIndex::new(storage, mode, config))),
             (768, "l2" | "euclidean") => IndexWrapper::L2Dim768(Arc::new(HnswIndex::new(storage, mode, config))),
             (1024, "l2" | "euclidean") => IndexWrapper::L2Dim1024(Arc::new(HnswIndex::new(storage, mode, config))),
             (1536, "l2" | "euclidean") => IndexWrapper::L2Dim1536(Arc::new(HnswIndex::new(storage, mode, config))),
             (3072, "l2" | "euclidean") => IndexWrapper::L2Dim3072(Arc::new(HnswIndex::new(storage, mode, config))),
             // Cosine
             (384, "cosine") => IndexWrapper::CosineDim384(Arc::new(HnswIndex::new(storage, mode, config))),
             (768, "cosine") => IndexWrapper::CosineDim768(Arc::new(HnswIndex::new(storage, mode, config))),
             (1024, "cosine") => IndexWrapper::CosineDim1024(Arc::new(HnswIndex::new(storage, mode, config))),
             (1536, "cosine") => IndexWrapper::CosineDim1536(Arc::new(HnswIndex::new(storage, mode, config))),
             (3072, "cosine") => IndexWrapper::CosineDim3072(Arc::new(HnswIndex::new(storage, mode, config))),
             // Poincare
             (384, "poincare") => IndexWrapper::PoincareDim384(Arc::new(HnswIndex::new(storage, mode, config))),
             (768, "poincare") => IndexWrapper::PoincareDim768(Arc::new(HnswIndex::new(storage, mode, config))),
             (1024, "poincare") => IndexWrapper::PoincareDim1024(Arc::new(HnswIndex::new(storage, mode, config))),
             (1536, "poincare") => IndexWrapper::PoincareDim1536(Arc::new(HnswIndex::new(storage, mode, config))),
             (3072, "poincare") => IndexWrapper::PoincareDim3072(Arc::new(HnswIndex::new(storage, mode, config))),
             // Lorentz
             (384, "lorentz") => IndexWrapper::LorentzDim384(Arc::new(HnswIndex::new(storage, mode, config))),
             (768, "lorentz") => IndexWrapper::LorentzDim768(Arc::new(HnswIndex::new(storage, mode, config))),
             (1024, "lorentz") => IndexWrapper::LorentzDim1024(Arc::new(HnswIndex::new(storage, mode, config))),
             (1536, "lorentz") => IndexWrapper::LorentzDim1536(Arc::new(HnswIndex::new(storage, mode, config))),
             (3072, "lorentz") => IndexWrapper::LorentzDim3072(Arc::new(HnswIndex::new(storage, mode, config))),

             _ => return Err(JsValue::from_str(&format!("Unsupported config: dim={dimension}, metric={metric}. Supported dims: 384, 768, 1024, 1536, 3072. Supported metrics: l2, euclidean, cosine, poincare, lorentz"))),
        };

        Ok(Self {
            index,
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            dimension,
        })
    }

    /// Inserts a vector.
    ///
    /// # Errors
    /// Returns error on dimension mismatch or duplicate ID.
    pub fn insert(&self, id: u32, vector: &[f64]) -> Result<(), JsValue> {
        if vector.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Dimension mismatch: expected {}.",
                self.dimension
            )));
        }

        let mut id_map = self.id_map.write();
        let mut rev_map = self.rev_map.write();

        if id_map.contains_key(&id) {
            return Err(JsValue::from_str("Duplicate ID not supported"));
        }

        macro_rules! insert_impl {
            ($idx:expr) => {
                $idx.insert(vector, HashMap::new())
                    .map_err(|e| JsValue::from_str(&e))?
            };
        }

        let internal_id = match &self.index {
            IndexWrapper::L2Dim384(idx) => insert_impl!(idx),
            IndexWrapper::L2Dim768(idx) => insert_impl!(idx),
            IndexWrapper::L2Dim1024(idx) => insert_impl!(idx),
            IndexWrapper::L2Dim1536(idx) => insert_impl!(idx),
            IndexWrapper::L2Dim3072(idx) => insert_impl!(idx),
            IndexWrapper::CosineDim384(idx) => insert_impl!(idx),
            IndexWrapper::CosineDim768(idx) => insert_impl!(idx),
            IndexWrapper::CosineDim1024(idx) => insert_impl!(idx),
            IndexWrapper::CosineDim1536(idx) => insert_impl!(idx),
            IndexWrapper::CosineDim3072(idx) => insert_impl!(idx),
            IndexWrapper::PoincareDim384(idx) => insert_impl!(idx),
            IndexWrapper::PoincareDim768(idx) => insert_impl!(idx),
            IndexWrapper::PoincareDim1024(idx) => insert_impl!(idx),
            IndexWrapper::PoincareDim1536(idx) => insert_impl!(idx),
            IndexWrapper::PoincareDim3072(idx) => insert_impl!(idx),
            IndexWrapper::LorentzDim384(idx) => insert_impl!(idx),
            IndexWrapper::LorentzDim768(idx) => insert_impl!(idx),
            IndexWrapper::LorentzDim1024(idx) => insert_impl!(idx),
            IndexWrapper::LorentzDim1536(idx) => insert_impl!(idx),
            IndexWrapper::LorentzDim3072(idx) => insert_impl!(idx),
        };

        id_map.insert(id, internal_id);
        rev_map.insert(internal_id, id);

        Ok(())
    }

    /// Searches for nearest neighbors.
    ///
    /// # Errors
    /// Returns error on dimension mismatch.
    pub fn search(&self, vector: &[f64], k: usize) -> Result<JsValue, JsValue> {
        if vector.len() != self.dimension {
            return Err(JsValue::from_str("Dimension mismatch"));
        }

        macro_rules! search_impl {
            ($idx:expr) => {
                $idx.search(vector, k, 100, &HashMap::new(), &[], None, None)
            };
        }

        let results = match &self.index {
            IndexWrapper::L2Dim384(idx) => search_impl!(idx),
            IndexWrapper::L2Dim768(idx) => search_impl!(idx),
            IndexWrapper::L2Dim1024(idx) => search_impl!(idx),
            IndexWrapper::L2Dim1536(idx) => search_impl!(idx),
            IndexWrapper::L2Dim3072(idx) => search_impl!(idx),
            IndexWrapper::CosineDim384(idx) => search_impl!(idx),
            IndexWrapper::CosineDim768(idx) => search_impl!(idx),
            IndexWrapper::CosineDim1024(idx) => search_impl!(idx),
            IndexWrapper::CosineDim1536(idx) => search_impl!(idx),
            IndexWrapper::CosineDim3072(idx) => search_impl!(idx),
            IndexWrapper::PoincareDim384(idx) => search_impl!(idx),
            IndexWrapper::PoincareDim768(idx) => search_impl!(idx),
            IndexWrapper::PoincareDim1024(idx) => search_impl!(idx),
            IndexWrapper::PoincareDim1536(idx) => search_impl!(idx),
            IndexWrapper::PoincareDim3072(idx) => search_impl!(idx),
            IndexWrapper::LorentzDim384(idx) => search_impl!(idx),
            IndexWrapper::LorentzDim768(idx) => search_impl!(idx),
            IndexWrapper::LorentzDim1024(idx) => search_impl!(idx),
            IndexWrapper::LorentzDim1536(idx) => search_impl!(idx),
            IndexWrapper::LorentzDim3072(idx) => search_impl!(idx),
        };

        let rev_map = self.rev_map.read();

        let mapped: Vec<serde_json::Value> = results
            .iter()
            .map(|(internal_id, dist)| {
                let user_id = rev_map.get(internal_id).copied().unwrap_or(*internal_id);
                serde_json::json!({
                    "id": user_id,
                    "distance": dist
                })
            })
            .collect();

        Ok(serde_wasm_bindgen::to_value(&mapped)?)
    }

    /// Persist current state to `IndexedDB`.
    ///
    /// # Errors
    /// Returns error if `IndexedDB` operations fail.
    pub async fn save(&self) -> Result<(), JsValue> {
        let rexie = Rexie::builder(DB_NAME)
            .version(1)
            .add_object_store(ObjectStore::new(STORE_NAME))
            .build()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let transaction = rexie
            .transaction(&[STORE_NAME], TransactionMode::ReadWrite)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let db_store = transaction
            .store(STORE_NAME)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // 1. Export Storage (Bytes)
        let vector_store = match &self.index {
            IndexWrapper::L2Dim384(idx) => idx.get_storage(),
            IndexWrapper::L2Dim768(idx) => idx.get_storage(),
            IndexWrapper::L2Dim1024(idx) => idx.get_storage(),
            IndexWrapper::L2Dim1536(idx) => idx.get_storage(),
            IndexWrapper::L2Dim3072(idx) => idx.get_storage(),
            IndexWrapper::CosineDim384(idx) => idx.get_storage(),
            IndexWrapper::CosineDim768(idx) => idx.get_storage(),
            IndexWrapper::CosineDim1024(idx) => idx.get_storage(),
            IndexWrapper::CosineDim1536(idx) => idx.get_storage(),
            IndexWrapper::CosineDim3072(idx) => idx.get_storage(),
            IndexWrapper::PoincareDim384(idx) => idx.get_storage(),
            IndexWrapper::PoincareDim768(idx) => idx.get_storage(),
            IndexWrapper::PoincareDim1024(idx) => idx.get_storage(),
            IndexWrapper::PoincareDim1536(idx) => idx.get_storage(),
            IndexWrapper::PoincareDim3072(idx) => idx.get_storage(),
            IndexWrapper::LorentzDim384(idx) => idx.get_storage(),
            IndexWrapper::LorentzDim768(idx) => idx.get_storage(),
            IndexWrapper::LorentzDim1024(idx) => idx.get_storage(),
            IndexWrapper::LorentzDim1536(idx) => idx.get_storage(),
            IndexWrapper::LorentzDim3072(idx) => idx.get_storage(),
        };

        let store_bytes = vector_store.as_ref().export();
        let store_js = serde_wasm_bindgen::to_value(&store_bytes)?;
        db_store
            .put(&store_js, Some(&JsValue::from_str("vectors")))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // 2. Export Index (Bytes)
        macro_rules! save_impl {
            ($idx:expr) => {
                $idx.save_to_bytes().map_err(|e| JsValue::from_str(&e))?
            };
        }

        let index_bytes = match &self.index {
            IndexWrapper::L2Dim384(idx) => save_impl!(idx),
            IndexWrapper::L2Dim768(idx) => save_impl!(idx),
            IndexWrapper::L2Dim1024(idx) => save_impl!(idx),
            IndexWrapper::L2Dim1536(idx) => save_impl!(idx),
            IndexWrapper::L2Dim3072(idx) => save_impl!(idx),
            IndexWrapper::CosineDim384(idx) => save_impl!(idx),
            IndexWrapper::CosineDim768(idx) => save_impl!(idx),
            IndexWrapper::CosineDim1024(idx) => save_impl!(idx),
            IndexWrapper::CosineDim1536(idx) => save_impl!(idx),
            IndexWrapper::CosineDim3072(idx) => save_impl!(idx),
            IndexWrapper::PoincareDim384(idx) => save_impl!(idx),
            IndexWrapper::PoincareDim768(idx) => save_impl!(idx),
            IndexWrapper::PoincareDim1024(idx) => save_impl!(idx),
            IndexWrapper::PoincareDim1536(idx) => save_impl!(idx),
            IndexWrapper::PoincareDim3072(idx) => save_impl!(idx),
            IndexWrapper::LorentzDim384(idx) => save_impl!(idx),
            IndexWrapper::LorentzDim768(idx) => save_impl!(idx),
            IndexWrapper::LorentzDim1024(idx) => save_impl!(idx),
            IndexWrapper::LorentzDim1536(idx) => save_impl!(idx),
            IndexWrapper::LorentzDim3072(idx) => save_impl!(idx),
        };
        let index_js = serde_wasm_bindgen::to_value(&index_bytes)?;
        db_store
            .put(&index_js, Some(&JsValue::from_str("index")))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // 3. Export ID Maps
        // Important: Serialize *before* awaiting to drop the lock!
        let map_js = {
            let id_map = self.id_map.read();
            serde_wasm_bindgen::to_value(&*id_map)?
        };

        db_store
            .put(&map_js, Some(&JsValue::from_str("id_map")))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        transaction
            .done()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        log("Saved to IndexedDB");
        Ok(())
    }

    /// Load state from `IndexedDB`.
    ///
    /// # Errors
    /// Returns error if `IndexedDB` operations fail.
    pub async fn load(&mut self) -> Result<bool, JsValue> {
        let rexie = Rexie::builder(DB_NAME)
            .version(1)
            .add_object_store(ObjectStore::new(STORE_NAME))
            .build()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let transaction = rexie
            .transaction(&[STORE_NAME], TransactionMode::ReadOnly)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let db_store = transaction
            .store(STORE_NAME)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Retrieve Vectors
        let vectors_js = db_store
            .get(&JsValue::from_str("vectors"))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if vectors_js.is_undefined() {
            return Ok(false);
        }

        let vectors_bytes: Vec<u8> = serde_wasm_bindgen::from_value(vectors_js)?;

        // Retrieve Index
        let index_js = db_store
            .get(&JsValue::from_str("index"))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let index_bytes: Vec<u8> = serde_wasm_bindgen::from_value(index_js)?;

        // Retrieve ID Map
        let map_js = db_store
            .get(&JsValue::from_str("id_map"))
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let id_map_data: HashMap<u32, u32> = serde_wasm_bindgen::from_value(map_js)?;

        // Reconstruct
        let element_size = self.dimension * 4;
        let storage = Arc::new(VectorStore::from_bytes(
            std::path::Path::new("mem"),
            element_size,
            &vectors_bytes,
        ));

        let config = Arc::new(GlobalConfig::default());
        let mode = QuantizationMode::None;

        // 2. Restore Index
        // We match on self.index to determine which type to load into
        macro_rules! load_impl {
            ($variant:ident) => {
                IndexWrapper::$variant(Arc::new(
                    HnswIndex::load_from_bytes(&index_bytes, storage, mode, config)
                        .map_err(|e| JsValue::from_str(&e))?,
                ))
            };
        }

        let new_index_wrapper = match &self.index {
            IndexWrapper::L2Dim384(_) => load_impl!(L2Dim384),
            IndexWrapper::L2Dim768(_) => load_impl!(L2Dim768),
            IndexWrapper::L2Dim1024(_) => load_impl!(L2Dim1024),
            IndexWrapper::L2Dim1536(_) => load_impl!(L2Dim1536),
            IndexWrapper::L2Dim3072(_) => load_impl!(L2Dim3072),
            IndexWrapper::CosineDim384(_) => load_impl!(CosineDim384),
            IndexWrapper::CosineDim768(_) => load_impl!(CosineDim768),
            IndexWrapper::CosineDim1024(_) => load_impl!(CosineDim1024),
            IndexWrapper::CosineDim1536(_) => load_impl!(CosineDim1536),
            IndexWrapper::CosineDim3072(_) => load_impl!(CosineDim3072),
            IndexWrapper::PoincareDim384(_) => load_impl!(PoincareDim384),
            IndexWrapper::PoincareDim768(_) => load_impl!(PoincareDim768),
            IndexWrapper::PoincareDim1024(_) => load_impl!(PoincareDim1024),
            IndexWrapper::PoincareDim1536(_) => load_impl!(PoincareDim1536),
            IndexWrapper::PoincareDim3072(_) => load_impl!(PoincareDim3072),
            IndexWrapper::LorentzDim384(_) => load_impl!(LorentzDim384),
            IndexWrapper::LorentzDim768(_) => load_impl!(LorentzDim768),
            IndexWrapper::LorentzDim1024(_) => load_impl!(LorentzDim1024),
            IndexWrapper::LorentzDim1536(_) => load_impl!(LorentzDim1536),
            IndexWrapper::LorentzDim3072(_) => load_impl!(LorentzDim3072),
        };

        // Update self
        self.index = new_index_wrapper;

        // Update Maps
        let mut id_map = self.id_map.write();
        let mut rev_map = self.rev_map.write();

        id_map.clone_from(&id_map_data);

        rev_map.clear();
        for (k, v) in id_map_data {
            rev_map.insert(v, k);
        }

        log("Loaded from IndexedDB");
        Ok(true)
    }
}
