// ────────────────────────────────────────────────────────────────
//  nietzsche-embed  — unit tests (Sprint 3)
//
//  Strategy:
//    • OnnxVectorizer requires real ONNX models — cannot be unit-tested
//      without fixtures, so we test the pure-logic helpers (normalize,
//      mean_pooling) through a standalone reimplementation that mirrors
//      the production code.
//    • RemoteVectorizer is tested for construction, provider parsing,
//      dimension reporting, and Gemini-not-implemented error.
//    • Metric enum coverage, ApiProvider::FromStr edge cases.
// ────────────────────────────────────────────────────────────────

use nietzsche_embed::{ApiProvider, Metric, RemoteVectorizer, Vectorizer};
use std::str::FromStr;

// ═══════════════════════════════════════════════════════════════
// 1. Metric enum
// ═══════════════════════════════════════════════════════════════

#[test]
fn metric_variants_exist() {
    let _p = Metric::Poincare;
    let _l = Metric::L2;
    let _c = Metric::Cosine;
    let _n = Metric::None;
}

#[test]
fn metric_clone_and_copy() {
    let m = Metric::Poincare;
    let m2 = m; // Copy
    let m3 = m.clone();
    assert_eq!(m, m2);
    assert_eq!(m, m3);
}

#[test]
fn metric_debug_format() {
    assert_eq!(format!("{:?}", Metric::Poincare), "Poincare");
    assert_eq!(format!("{:?}", Metric::L2), "L2");
    assert_eq!(format!("{:?}", Metric::Cosine), "Cosine");
    assert_eq!(format!("{:?}", Metric::None), "None");
}

#[test]
fn metric_equality() {
    assert_eq!(Metric::Poincare, Metric::Poincare);
    assert_ne!(Metric::Poincare, Metric::L2);
    assert_ne!(Metric::L2, Metric::Cosine);
    assert_ne!(Metric::Cosine, Metric::None);
}

// ═══════════════════════════════════════════════════════════════
// 2. ApiProvider enum — FromStr
// ═══════════════════════════════════════════════════════════════

#[test]
fn api_provider_from_str_lowercase() {
    assert_eq!(ApiProvider::from_str("openai").unwrap(), ApiProvider::OpenAI);
    assert_eq!(ApiProvider::from_str("cohere").unwrap(), ApiProvider::Cohere);
    assert_eq!(ApiProvider::from_str("voyage").unwrap(), ApiProvider::Voyage);
    assert_eq!(
        ApiProvider::from_str("mistral").unwrap(),
        ApiProvider::Mistral
    );
    assert_eq!(ApiProvider::from_str("gemini").unwrap(), ApiProvider::Gemini);
    assert_eq!(
        ApiProvider::from_str("openrouter").unwrap(),
        ApiProvider::OpenRouter
    );
    assert_eq!(
        ApiProvider::from_str("generic").unwrap(),
        ApiProvider::Generic
    );
}

#[test]
fn api_provider_from_str_case_insensitive() {
    assert_eq!(ApiProvider::from_str("OpenAI").unwrap(), ApiProvider::OpenAI);
    assert_eq!(
        ApiProvider::from_str("COHERE").unwrap(),
        ApiProvider::Cohere
    );
    assert_eq!(
        ApiProvider::from_str("Voyage").unwrap(),
        ApiProvider::Voyage
    );
    assert_eq!(
        ApiProvider::from_str("MISTRAL").unwrap(),
        ApiProvider::Mistral
    );
    assert_eq!(
        ApiProvider::from_str("GeMiNi").unwrap(),
        ApiProvider::Gemini
    );
    assert_eq!(
        ApiProvider::from_str("OpenRouter").unwrap(),
        ApiProvider::OpenRouter
    );
    assert_eq!(
        ApiProvider::from_str("GENERIC").unwrap(),
        ApiProvider::Generic
    );
}

#[test]
fn api_provider_from_str_invalid() {
    assert!(ApiProvider::from_str("").is_err());
    assert!(ApiProvider::from_str("gpt4").is_err());
    assert!(ApiProvider::from_str("anthropic").is_err());
    assert!(ApiProvider::from_str("NietzscheDB").is_err());
    assert!(ApiProvider::from_str("open_ai").is_err()); // underscore
}

#[test]
fn api_provider_clone_and_eq() {
    let a = ApiProvider::OpenAI;
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn api_provider_debug_format() {
    assert_eq!(format!("{:?}", ApiProvider::OpenAI), "OpenAI");
    assert_eq!(format!("{:?}", ApiProvider::Cohere), "Cohere");
    assert_eq!(format!("{:?}", ApiProvider::Voyage), "Voyage");
    assert_eq!(format!("{:?}", ApiProvider::Mistral), "Mistral");
    assert_eq!(format!("{:?}", ApiProvider::Gemini), "Gemini");
    assert_eq!(format!("{:?}", ApiProvider::OpenRouter), "OpenRouter");
    assert_eq!(format!("{:?}", ApiProvider::Generic), "Generic");
}

#[test]
fn api_provider_all_variants_distinct() {
    let providers = [
        ApiProvider::OpenAI,
        ApiProvider::Cohere,
        ApiProvider::Voyage,
        ApiProvider::Mistral,
        ApiProvider::Gemini,
        ApiProvider::OpenRouter,
        ApiProvider::Generic,
    ];
    for i in 0..providers.len() {
        for j in (i + 1)..providers.len() {
            assert_ne!(providers[i], providers[j]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 3. RemoteVectorizer — construction and trait
// ═══════════════════════════════════════════════════════════════

#[test]
fn remote_vectorizer_new_openai() {
    let v = RemoteVectorizer::new(
        ApiProvider::OpenAI,
        "sk-test-key".to_string(),
        "text-embedding-3-small".to_string(),
        None,
    );
    // dimension() returns 0 for remote APIs
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_new_cohere() {
    let v = RemoteVectorizer::new(
        ApiProvider::Cohere,
        "key123".to_string(),
        "embed-english-v3.0".to_string(),
        None,
    );
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_new_voyage() {
    let v = RemoteVectorizer::new(
        ApiProvider::Voyage,
        "key456".to_string(),
        "voyage-2".to_string(),
        None,
    );
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_new_mistral() {
    let v = RemoteVectorizer::new(
        ApiProvider::Mistral,
        "key789".to_string(),
        "mistral-embed".to_string(),
        None,
    );
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_new_openrouter() {
    let v = RemoteVectorizer::new(
        ApiProvider::OpenRouter,
        "keyOR".to_string(),
        "openai/text-embedding-3-small".to_string(),
        None,
    );
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_new_generic() {
    let v = RemoteVectorizer::new(
        ApiProvider::Generic,
        "keyGEN".to_string(),
        "custom-model".to_string(),
        Some("http://localhost:8080/embeddings".to_string()),
    );
    assert_eq!(v.dimension(), 0);
}

#[test]
fn remote_vectorizer_custom_base_url() {
    // Ensures custom base_url can be provided for any provider
    let _v = RemoteVectorizer::new(
        ApiProvider::OpenAI,
        "key".to_string(),
        "model".to_string(),
        Some("http://my-proxy.local:3000/v1/embeddings".to_string()),
    );
}

#[tokio::test]
async fn remote_vectorizer_gemini_not_implemented() {
    let v = RemoteVectorizer::new(
        ApiProvider::Gemini,
        "key".to_string(),
        "text-embedding-004".to_string(),
        None,
    );
    let result = v
        .vectorize(vec!["hello".to_string()])
        .await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Gemini embedding not yet implemented"),
        "Expected Gemini not-implemented error, got: {err_msg}"
    );
}

#[tokio::test]
async fn remote_vectorizer_gemini_suggests_generic() {
    let v = RemoteVectorizer::new(
        ApiProvider::Gemini,
        "key".to_string(),
        "model".to_string(),
        None,
    );
    let err = v.vectorize(vec!["test".to_string()]).await.unwrap_err();
    assert!(
        err.to_string().contains("use Generic if compatible"),
        "Error should suggest using Generic provider"
    );
}

// ═══════════════════════════════════════════════════════════════
// 4. Normalization logic (mirror of OnnxVectorizer::normalize)
//    Tested via standalone functions since OnnxVectorizer requires
//    actual ONNX model files for construction.
// ═══════════════════════════════════════════════════════════════

fn normalize_poincare(vec: &mut [f64]) {
    const EPSILON: f64 = 1e-5;
    let norm_sq: f64 = vec.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm >= 1.0 {
        let scale = (1.0 - EPSILON) / (norm + 1e-12);
        for x in vec.iter_mut() {
            *x *= scale;
        }
    }
}

fn normalize_l2(vec: &mut [f64]) {
    let norm_sq: f64 = vec.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

fn normalize_none(vec: &mut [f64]) {
    // no-op — identical to Metric::None branch
    let _ = vec;
}

#[test]
fn normalize_poincare_inside_ball_untouched() {
    let mut v = vec![0.3, 0.4]; // norm = 0.5 < 1.0
    let original = v.clone();
    normalize_poincare(&mut v);
    assert_eq!(v, original); // no change
}

#[test]
fn normalize_poincare_on_boundary_rescaled() {
    let mut v = vec![0.6, 0.8]; // norm = 1.0 exactly
    normalize_poincare(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1.0, "norm should be < 1 after Poincaré normalization, got {norm}");
}

#[test]
fn normalize_poincare_outside_ball_rescaled() {
    let mut v = vec![1.5, 2.0]; // norm = 2.5
    normalize_poincare(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1.0, "norm should be < 1, got {norm}");
    assert!(
        norm > 0.999,
        "should be close to boundary, got {norm}"
    );
}

#[test]
fn normalize_poincare_preserves_direction() {
    let mut v = vec![3.0, 4.0]; // norm = 5.0
    let ratio_before = v[0] / v[1];
    normalize_poincare(&mut v);
    let ratio_after = v[0] / v[1];
    assert!(
        (ratio_before - ratio_after).abs() < 1e-10,
        "direction should be preserved"
    );
}

#[test]
fn normalize_poincare_zero_vector() {
    let mut v = vec![0.0, 0.0, 0.0];
    normalize_poincare(&mut v); // norm = 0 < 1.0 → no change
    assert_eq!(v, vec![0.0, 0.0, 0.0]);
}

#[test]
fn normalize_poincare_high_dim() {
    // 1024-dim vector with each coord = 0.1 → norm ≈ 3.2
    let mut v = vec![0.1; 1024];
    normalize_poincare(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1.0);
}

#[test]
fn normalize_l2_unit_vector_stays() {
    let mut v = vec![1.0, 0.0, 0.0];
    normalize_l2(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-10);
    assert!(v[1].abs() < 1e-10);
    assert!(v[2].abs() < 1e-10);
}

#[test]
fn normalize_l2_scales_to_unit() {
    let mut v = vec![3.0, 4.0]; // norm = 5.0
    normalize_l2(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "should be unit norm, got {norm}"
    );
    assert!((v[0] - 0.6).abs() < 1e-10);
    assert!((v[1] - 0.8).abs() < 1e-10);
}

#[test]
fn normalize_l2_zero_vector_stays_zero() {
    let mut v = vec![0.0, 0.0];
    normalize_l2(&mut v);
    assert_eq!(v, vec![0.0, 0.0]);
}

#[test]
fn normalize_l2_negative_coords() {
    let mut v = vec![-3.0, 4.0];
    normalize_l2(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
    assert!(v[0] < 0.0); // sign preserved
}

#[test]
fn normalize_none_leaves_unchanged() {
    let mut v = vec![42.0, -99.5, 0.001];
    let original = v.clone();
    normalize_none(&mut v);
    assert_eq!(v, original);
}

// ═══════════════════════════════════════════════════════════════
// 5. Mean pooling logic (mirror of OnnxVectorizer::mean_pooling)
// ═══════════════════════════════════════════════════════════════

fn mean_pooling(
    hidden_state: &[f32],  // flattened [batch, seq, hidden]
    attention_mask: &[i64], // flattened [batch, seq]
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<Vec<f64>> {
    let mut output = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let mut sum_vec = vec![0.0f64; hidden_dim];
        let mut count = 0.0f64;
        for j in 0..seq_len {
            if attention_mask[i * seq_len + j] == 1 {
                for k in 0..hidden_dim {
                    sum_vec[k] += f64::from(hidden_state[i * seq_len * hidden_dim + j * hidden_dim + k]);
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for item in sum_vec.iter_mut() {
                *item /= count;
            }
        }
        output.push(sum_vec);
    }
    output
}

#[test]
fn mean_pooling_single_token() {
    // batch=1, seq=1, hidden=3, all attended
    let hidden = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1i64];
    let result = mean_pooling(&hidden, &mask, 1, 1, 3);
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 1.0).abs() < 1e-10);
    assert!((result[0][1] - 2.0).abs() < 1e-10);
    assert!((result[0][2] - 3.0).abs() < 1e-10);
}

#[test]
fn mean_pooling_two_tokens_averaged() {
    // batch=1, seq=2, hidden=2
    let hidden = vec![2.0f32, 4.0, 6.0, 8.0];
    let mask = vec![1i64, 1]; // both attended
    let result = mean_pooling(&hidden, &mask, 1, 2, 2);
    assert!((result[0][0] - 4.0).abs() < 1e-10); // (2+6)/2
    assert!((result[0][1] - 6.0).abs() < 1e-10); // (4+8)/2
}

#[test]
fn mean_pooling_masked_tokens_ignored() {
    // batch=1, seq=3, hidden=2
    // token 0: attended, token 1: masked, token 2: attended
    let hidden = vec![
        1.0f32, 2.0, // token 0
        100.0, 200.0, // token 1 (should be ignored)
        3.0, 4.0, // token 2
    ];
    let mask = vec![1i64, 0, 1];
    let result = mean_pooling(&hidden, &mask, 1, 3, 2);
    assert!((result[0][0] - 2.0).abs() < 1e-10); // (1+3)/2
    assert!((result[0][1] - 3.0).abs() < 1e-10); // (2+4)/2
}

#[test]
fn mean_pooling_all_masked() {
    // batch=1, seq=2, hidden=2, no tokens attended
    let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0i64, 0];
    let result = mean_pooling(&hidden, &mask, 1, 2, 2);
    // count = 0 → no division → zeros
    assert_eq!(result[0], vec![0.0, 0.0]);
}

#[test]
fn mean_pooling_multi_batch() {
    // batch=2, seq=2, hidden=2
    let hidden = vec![
        // batch 0
        1.0f32, 0.0, // token 0
        3.0, 0.0, // token 1
        // batch 1
        0.0, 10.0, // token 0
        0.0, 20.0, // token 1
    ];
    let mask = vec![
        1i64, 1, // batch 0: both attended
        1, 1, // batch 1: both attended
    ];
    let result = mean_pooling(&hidden, &mask, 2, 2, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0][0] - 2.0).abs() < 1e-10); // (1+3)/2
    assert!((result[1][1] - 15.0).abs() < 1e-10); // (10+20)/2
}

#[test]
fn mean_pooling_f32_to_f64_precision() {
    // Ensures f32 → f64 promotion
    let hidden = vec![0.1f32, 0.2];
    let mask = vec![1i64];
    let result = mean_pooling(&hidden, &mask, 1, 1, 2);
    // f32 0.1 → f64 should be close to 0.1 but not exact
    assert!((result[0][0] - 0.1).abs() < 1e-6);
    assert!((result[0][1] - 0.2).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════
// 6. Vectorizer trait — object safety
// ═══════════════════════════════════════════════════════════════

#[test]
fn vectorizer_trait_is_object_safe() {
    // RemoteVectorizer implements Vectorizer — can be used as dyn
    let v: Box<dyn Vectorizer> = Box::new(RemoteVectorizer::new(
        ApiProvider::OpenAI,
        "key".to_string(),
        "model".to_string(),
        None,
    ));
    assert_eq!(v.dimension(), 0);
}

#[tokio::test]
async fn vectorizer_trait_dispatch_via_dyn() {
    let v: Box<dyn Vectorizer> = Box::new(RemoteVectorizer::new(
        ApiProvider::Gemini,
        "key".to_string(),
        "model".to_string(),
        None,
    ));
    // Gemini returns error — but we can call via trait object
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════
// 7. RemoteVectorizer — provider URL defaults
// ═══════════════════════════════════════════════════════════════

// We can't directly test private URL logic, but we can verify that
// vectorize() fails gracefully with connection errors (no real API)
// for each provider when called with empty texts.

#[tokio::test]
async fn remote_vectorizer_openai_empty_texts() {
    let v = RemoteVectorizer::new(
        ApiProvider::OpenAI,
        "fake-key".to_string(),
        "text-embedding-3-small".to_string(),
        Some("http://127.0.0.1:1/v1/embeddings".to_string()), // unreachable
    );
    let result = v.vectorize(vec![]).await;
    // Empty texts still makes an API call with empty array
    // Will fail with connection error (port 1 unreachable)
    assert!(result.is_err());
}

#[tokio::test]
async fn remote_vectorizer_cohere_connection_error() {
    let v = RemoteVectorizer::new(
        ApiProvider::Cohere,
        "fake-key".to_string(),
        "embed-english-v3.0".to_string(),
        Some("http://127.0.0.1:1/v1/embed".to_string()),
    );
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn remote_vectorizer_voyage_connection_error() {
    let v = RemoteVectorizer::new(
        ApiProvider::Voyage,
        "fake-key".to_string(),
        "voyage-2".to_string(),
        Some("http://127.0.0.1:1/v1/embeddings".to_string()),
    );
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn remote_vectorizer_mistral_connection_error() {
    let v = RemoteVectorizer::new(
        ApiProvider::Mistral,
        "fake-key".to_string(),
        "mistral-embed".to_string(),
        Some("http://127.0.0.1:1/v1/embeddings".to_string()),
    );
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn remote_vectorizer_openrouter_connection_error() {
    let v = RemoteVectorizer::new(
        ApiProvider::OpenRouter,
        "fake-key".to_string(),
        "model".to_string(),
        Some("http://127.0.0.1:1/v1/embeddings".to_string()),
    );
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn remote_vectorizer_generic_connection_error() {
    let v = RemoteVectorizer::new(
        ApiProvider::Generic,
        "fake-key".to_string(),
        "custom-model".to_string(),
        Some("http://127.0.0.1:1/embed".to_string()),
    );
    let result = v.vectorize(vec!["test".to_string()]).await;
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════
// 8. Poincaré boundary math — epsilon invariants
// ═══════════════════════════════════════════════════════════════

#[test]
fn poincare_epsilon_is_1e5() {
    // After normalization, max norm = 1 - 1e-5
    let mut v = vec![10.0, 0.0]; // norm = 10
    normalize_poincare(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - (1.0 - 1e-5)).abs() < 1e-8,
        "Norm should be exactly 1 - epsilon, got {norm}"
    );
}

#[test]
fn poincare_norm_0999_untouched() {
    let mut v = vec![0.999, 0.0]; // norm = 0.999 < 1.0
    let original = v.clone();
    normalize_poincare(&mut v);
    assert_eq!(v, original);
}

#[test]
fn poincare_norm_exactly_1_rescaled() {
    let mut v = vec![0.6, 0.8]; // norm = 1.0
    normalize_poincare(&mut v);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1.0);
}

// ═══════════════════════════════════════════════════════════════
// 9. Stress: large batch normalization
// ═══════════════════════════════════════════════════════════════

#[test]
fn normalize_poincare_batch_100_vectors() {
    for i in 0..100 {
        let scale = (i + 1) as f64 * 0.1; // 0.1 .. 10.0
        let mut v = vec![scale, 0.0, 0.0];
        normalize_poincare(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "vector {i} norm = {norm} >= 1.0");
    }
}

#[test]
fn normalize_l2_batch_100_vectors() {
    for i in 1..=100 {
        let scale = i as f64 * 0.5;
        let mut v = vec![scale, scale * 2.0];
        normalize_l2(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "vector {i} norm = {norm}"
        );
    }
}
