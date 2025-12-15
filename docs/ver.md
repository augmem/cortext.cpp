# **Engineering Task: Implement Audio Emotion Recognition (VER) via ImageBind**

## **Objective**

Enable real-time Vocal Emotion Recognition (VER) by comparing incoming audio embeddings against pre-calculated "Emotion Centroids."  
Constraint: All embeddings must be 256d (matching our production storage schema).

## **Part 1: Dataset Preparation (Offline)**

### **1\. Acquire Data**

Download the **RAVDESS (Speech Only)** dataset.

* **Source:** Ryerson Audio-Visual Database of Emotional Speech and Song.  
* **Files:** You only need the Audio\_Speech\_Actors\_01-24.zip. Do not use the "Song" files.

### **2\. Parse Filenames**

RAVDESS uses a numerical filename convention (e.g., 03-01-06-01-01-01-01.wav).  
The third number determines the emotion. Map them to our Cortext types as follows:

| RAVDESS Code | Emotion | Cortext Mapping | Notes |
| :---- | :---- | :---- | :---- |
| 01 | Neutral | **Neutral** | Combine with Calm for a robust baseline. |
| 02 | Calm | **Neutral** | Used in Neutral, but also used for **Love** mix. |
| 03 | Happy | **Joy** | Used for **Love** mix. |
| 04 | Sad | **Sadness** |  |
| 05 | Angry | **Anger** |  |
| 06 | Fearful | **Fear** |  |
| 07 | Disgust | *Skip* | Not in our 6-dim map. |
| 08 | Surprised | **Surprise** | Used for **Love** mix. |

### **Part 1.5: Synthetic "Love" Centroid**

Since standard datasets lack a "Love" category, we will synthesize it mathematically by mixing constituent emotions as requested.

**Logic:** Love $\\approx$ Joy (Positivity) \+ Calm (Tenderness) \+ Surprise (Engagement).

1. **Calculate Base Centroids First:** Compute the standard centroids for **Joy**, **Neutral** (using Calm/Neutral files), and **Surprise** normally.  
2. Compute Synthetic Vector:  
   After the base centroids are computed but before saving to JSON, calculate:$$V\_{love} \= \\frac{V\_{joy} \+ V\_{calm\\\_raw} \+ V\_{surprise}}{3}$$  
   *(Note: Ensure you keep the raw 'Calm' vector available for this calculation even if you merged it into 'Neutral' for the final map).*  
3. **Normalize:** L2 Normalize $V\_{love}$ to ensure it lies on the same hypersphere as the others.

### **3\. Generate Centroids (The "Bake")**

Update the existing script (scripts/generate\_centroid\_vectors.py) to perform the following:

1. **Iterate** through all RAVDESS files.  
2. **Filter** out Emotion 07 (Disgust).  
3. **Embed** each file:  
   * **CRITICAL:** Use the production ONNX ImageBind model. It is already truncated to 256d. Do not add an external projection layer.  
   * *Result:* (1, 256\) vector per file.  
4. **Normalize** every vector to unit length (L2 norm).  
5. **Group** vectors. Keep 02 (Calm), 03 (Joy), and 08 (Surprise) accessible for the synthetic step.  
6. **Calculate Mean:** Compute the average vector for each group.  
7. **Synthesize Love:** Perform the vector addition and normalization described in Part 1.5.  
8. **Export:** Save audio\_centroids.json (or .npy).  
   * Format: {"joy": \[...\], "anger": \[...\], "love": \[...\]}

## **Part 2: Runtime Implementation (Real-time)**

### **1\. Load Centroids**

On system startup (boot()), load the audio\_centroids.json file into memory as a dictionary or hash map.

### **2\. Classification Logic**

Inside the audio processing loop, perform the following operations on the user's incoming audio embedding (u\_audio\_256d) and the global Sensitivity knob (S):

1. Similarity Calculation:  
   Iterate through each loaded emotion centroid. Calculate the Dot Product (Cosine Similarity) between the normalized input embedding and the normalized centroid vector. Store these raw similarity scores as logits.  
2. Temperature Derivation:  
   Calculate the Softmax temperature ($\\beta$) using the global Sensitivity ($S$) knob.  
   * **Formula:** $\\beta \= 4.0 \+ (8.0 \\times S)$  
   * *Effect:* Higher Sensitivity results in a higher Beta, which sharpens the probability distribution (higher confidence). Lower Sensitivity results in a lower Beta, flattening the distribution (higher uncertainty).  
3. Probability Normalization:  
   Apply the Softmax function to the logits using the calculated temperature $\\beta$. This converts the raw similarity scores into a probability distribution that sums to 1.0.  
4. Output:  
   Return this probability vector (e.g., {'joy': 0.8, 'anger': 0.2, ...}) as the instantaneous vocal emotion signal ($e\_{audio}$).

### **3\. Integrate with Algorithm 4b (Mood)**

Pass the resulting probability vector as $e\_{audio}$ into the Mood Integrator.

* **Sanity Check:** If the user is silent or audio is noise, the embedding might drift. Ensure you have a Voice Activity Detector (VAD) gate before running this classification, or the system will hallucinate emotions from background noise.

## **Part 3: Validation**

1. **Self-Test:** Run the classifier against a held-out set of RAVDESS files.  
2. **Metric:** Check accuracy. If accuracy is \< 40%, the 256d truncation during ONNX export might have "crushed" the prosodic features.  
   * *Fix:* If this happens, we may need to train a lightweight linear classifier (logistic regression) on the 256d embeddings instead of using raw centroids, but try centroids first (cheaper).