//! Practical Example: Using Gumbel Reranking for RAG Training
//!
//! This example demonstrates a realistic RAG reranking training scenario:
//! 1. Retrieve candidate documents
//! 2. Score with reranker
//! 3. Generate soft attention mask using Gumbel
//! 4. Apply mask to attention (simulated)
//! 5. Compute language loss (simulated)
//! 6. Show how gradients would flow back to reranker
//!
//! Run with: cargo run --example gumbel_rag_training --features gumbel

#[cfg(not(feature = "gumbel"))]
fn main() {
    eprintln!("This example requires the 'gumbel' feature.");
    eprintln!("Run with: cargo run --example gumbel_rag_training --features gumbel");
}

#[cfg(feature = "gumbel")]
fn main() {
    use rank_relax::gumbel_attention_mask;
    use rand::{thread_rng, SeedableRng, Rng};
    use rand::rngs::StdRng;

    println!("RAG Reranking Training Simulation");
    println!("{}", "=".repeat(60));
    println!();

    // Simulate a realistic RAG scenario
    struct Query {
        text: &'static str,
        answer: &'static str,
    }

    struct Document {
        id: usize,
        text: &'static str,
        relevance: f64, // Ground truth (unknown during training)
    }

    let query = Query {
        text: "What is the capital of Turkey?",
        answer: "Ankara",
    };

    // Retrieved candidate documents (some relevant, some not)
    let documents = vec![
        Document { id: 1, text: "Ankara is the capital city of Turkey.", relevance: 1.0 },
        Document { id: 2, text: "Istanbul is the largest city in Turkey.", relevance: 0.5 },
        Document { id: 3, text: "Turkey is a country in Eurasia.", relevance: 0.3 },
        Document { id: 4, text: "The capital of France is Paris.", relevance: 0.0 },
        Document { id: 5, text: "Ankara became the capital in 1923.", relevance: 0.9 },
        Document { id: 6, text: "The weather in Turkey is generally warm.", relevance: 0.1 },
        Document { id: 7, text: "Ankara has a population of over 5 million.", relevance: 0.7 },
        Document { id: 8, text: "Turkey borders Greece and Bulgaria.", relevance: 0.2 },
    ];

    println!("Query: {}", query.text);
    println!("Expected Answer: {}", query.answer);
    println!();
    println!("Retrieved {} candidate documents:", documents.len());
    for doc in &documents {
        println!("  [doc{}] {} (relevance: {:.1})", doc.id, doc.text, doc.relevance);
    }
    println!();

    // Simulate reranker scores (initially imperfect, will improve with training)
    let mut reranker_scores = vec![
        0.65, // doc1: should be high (contains answer)
        0.55, // doc2: moderate
        0.40, // doc3: low
        0.30, // doc4: very low (irrelevant)
        0.70, // doc5: should be high (contains answer)
        0.25, // doc6: low
        0.60, // doc7: moderate-high
        0.35, // doc8: low
    ];

    println!("Initial Reranker Scores:");
    for (i, &score) in reranker_scores.iter().enumerate() {
        let doc = &documents[i];
        println!("  doc{}: {:.2} (true relevance: {:.1})", doc.id, score, doc.relevance);
    }
    println!();

    // Training loop simulation
    let k = 3; // Select top-3 documents
    let temperature = 0.5;
    let scale = 1.0;
    let learning_rate = 0.1;

    println!("Training Reranker ({} iterations, selecting top-{})", 5, k);
    println!("{}", "-".repeat(60));

    for iteration in 0..5 {
        let mut rng = StdRng::seed_from_u64(42 + iteration);
        
        // Generate soft attention mask
        let attention_mask = gumbel_attention_mask(
            &reranker_scores,
            k,
            temperature,
            scale,
            &mut rng,
        );

        // Simulate language model attention scores
        // (In real scenario, these come from LLM forward pass)
        let mut language_attention: Vec<f64> = documents.iter()
            .enumerate()
            .map(|(i, doc)| {
                // Simulate: documents containing answer get higher attention
                if doc.text.contains(&query.answer) {
                    0.7 + (i as f64 * 0.05)
                } else {
                    0.3 + (i as f64 * 0.02)
                }
            })
            .collect();

        // Apply mask to attention
        for i in 0..language_attention.len() {
            language_attention[i] *= attention_mask[i];
        }

        // Simulate language loss (lower = better)
        // In real scenario: loss = -log P(answer | query, selected_docs)
        let selected_attention: f64 = language_attention.iter().sum();
        let language_loss = 1.0 / (1.0 + selected_attention); // Inverse relationship

        // Simulate gradient update to reranker
        // (In real scenario, gradients flow: loss → attention → mask → reranker)
        for i in 0..reranker_scores.len() {
            // Gradient signal: if document helped reduce loss, increase its score
            let gradient_signal = if attention_mask[i] > 0.5 {
                // Selected document: update based on whether it helped
                if documents[i].relevance > 0.5 {
                    learning_rate * (1.0 - language_loss) // Good document: increase score
                } else {
                    -learning_rate * language_loss // Bad document: decrease score
                }
            } else {
                0.0 // Not selected: no update
            };
            
            reranker_scores[i] = (reranker_scores[i] + gradient_signal).max(0.0).min(1.0);
        }

        // Display iteration results
        println!("Iteration {}:", iteration + 1);
        println!("  Selected documents (mask > 0.5):");
        for (i, &mask) in attention_mask.iter().enumerate() {
            if mask > 0.5 {
                let doc = &documents[i];
                println!("    doc{}: mask={:.3}, relevance={:.1} - {}", 
                    doc.id, mask, doc.relevance, doc.text);
            }
        }
        println!("  Language loss: {:.4}", language_loss);
        println!("  Reranker scores updated");
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("Final Reranker Scores:");
    for (i, &score) in reranker_scores.iter().enumerate() {
        let doc = &documents[i];
        let improvement = score - reranker_scores[0];
        println!("  doc{}: {:.2} (true: {:.1}, improvement: {:+.2})", 
            doc.id, score, doc.relevance, improvement);
    }
    println!();

    // Check if reranker learned correctly
    let mut top_k_indices: Vec<usize> = (0..documents.len()).collect();
    top_k_indices.sort_by(|&a, &b| reranker_scores[b].partial_cmp(&reranker_scores[a]).unwrap());
    
    println!("Top-{} documents after training:", k);
    for (rank, &idx) in top_k_indices.iter().take(k).enumerate() {
        let doc = &documents[idx];
        println!("  {}. doc{}: score={:.2}, relevance={:.1} - {}", 
            rank + 1, doc.id, reranker_scores[idx], doc.relevance, doc.text);
    }
    println!();

    // Verify learning
    let top_k_avg_relevance: f64 = top_k_indices.iter()
        .take(k)
        .map(|&idx| documents[idx].relevance)
        .sum::<f64>() / k as f64;
    
    println!("Average relevance of top-{}: {:.2}", k, top_k_avg_relevance);
    if top_k_avg_relevance > 0.7 {
        println!("✅ Reranker successfully learned to select relevant documents!");
    } else {
        println!("⚠️  Reranker needs more training or parameter tuning");
    }
    println!();

    println!("Key Insights:");
    println!("  • Gumbel-Softmax enables differentiable document selection");
    println!("  • Gradients flow from language loss → mask → reranker");
    println!("  • No labeled relevance data needed (only query-answer pairs)");
    println!("  • Stochastic exploration helps find better document combinations");
}

