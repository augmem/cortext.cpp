# **Cortext Extension: Persistent Emotional State (The "Mood" Layer)**

## **1\. Conceptual Distinction**

To make the system behave like a biological entity rather than a database, we distinguish between:

* **Emotion (Phasic):** The immediate, fleeting reaction to a single signal (Algorithm 4). This represents the emotional content *of the event itself*.  
* **Mood (Tonic):** A persistent, slowly decaying background state that colors perception and retrieval. This represents the *internal state of the user*.

## **2\. The State Vector ($M\_t$)**

Instead of a single scalar, we maintain a **Mood Vector** $M\_t$ in the same semantic space as the 6-dimensional emotion projection map defined in §0.6 (Anger, Fear, Sadness, Joy, Love, Surprise).

$$M\_t \\in \\mathbb{R}^6$$

## **3\. Algorithm 4b: The Mood Integrator**

This algorithm runs *after* Algorithm 4 (Dynamic Update) but *before* Storage/Retrieval.

Sensor Input Definition ($e\_t$):  
The instantaneous emotion vector $e\_t$ is derived from available hardware sensors.

* **Current (Audio):** Vocal prosody (pitch variance), volume dynamics (shouting vs. whispering), and speech rate.  
* **Future (Bio):** Heart rate variability (HRV) and electrodermal activity (EDA) will be fused into this vector when sensors are available.

**Inputs:**

* $e\_t$: The instantaneous emotion vector from the current signal (Alg 4).  
* $S$: Sensitivity (Plasticity).  
* $T$: Stability (Inertia).  
* $M\_{t-1}$: Previous mood state.

**The Dynamics:**

1. **Reactivity (**$\\alpha$**):** Controlled by **Sensitivity (**$S$**)**. High $S$ makes the mood volatile and easily shifted by new events.$$\\alpha\_{mood}(S) \= \\text{lerp}(0.01, 0.20, S)$$  
2. **Decay (**$\\lambda$**):** Controlled by **Stability (**$T$**)**. High $T$ makes the mood linger longer (grudges, sustained depression, or lasting euphoria).$$\\lambda\_{mood}(T) \= \\text{lerp}(0.90, 0.999, T)$$

**Update Equation:**

$$M\_t \= \\underbrace{\\lambda\_{mood}(T) \\times M\_{t-1}}\_{\\text{Persistence}} \+ \\underbrace{\\alpha\_{mood}(S) \\times e\_t}\_{\\text{Reactivity}}$$  
*After update, we clamp or normalize* $M\_t$ *to ensure it doesn't explode.*

## **4\. Impact on Storage (Dual-Signal Metadata)**

To prevent false attribution (blaming a neutral person for an agitated mood), we explicitly separate the **Source** from the **Background** in the metadata.

Mechanism:  
When storing a memory $m$, we append two distinct vectors:

1. event\_emotion ($e\_t$): The emotion triggered specifically by *this* signal (e.g., laughter at a joke).  
2. ambient\_mood ($M\_t$): The background mood state at the timestamp (e.g., lingering agitation).

The Threshold Bias:  
The system uses the intensity of the Ambient Mood ($M\_t$) to bias the write threshold ($T\_{dynamic}$).

* *Rationale:* If the user is agitated, we *do* want to record neutral events (hyper-vigilance is a safety feature), even if we don't want to attribute the anger to them.

$$\\Delta T\_{mood} \= \- \\kappa \\times ||M\_t||$$

## **5\. Impact on Retrieval (Prosthetic Protocol)**

Safety Directive:  
For cognitive assistance, mood-congruent recall creates dangerous feedback loops. The system enforces a Strict Neutrality Protocol.

1. Force Neutral Ranking:  
   The retrieval ranking score is purely semantic. The mood bias weight is mathematically forced to zero.$$\\text{Score}(m) \= \\text{sim}(q, m) \\quad (\\text{i.e., } w\_{mood} \\equiv 0)$$  
2. Focus Spike:  
   Upon detection of an explicit query (e.g., "Who is this?"), the system momentarily forces Focus ($F$) to 1.0 to ensure objectivity.  
3. Context Display (Attribution Logic):  
   The system uses a heuristic to generate the grounding message, ensuring the user understands why the emotion exists.  
   * Case A: High Event Emotion ($e\_t \\gg 0$):  
     The emotion came from the interaction.  
     Msg \= "Context: You reacted with \[Emotion\] to this specific event."  
   * Case B: High Ambient Mood, Neutral Event ($M\_t \\gg 0, e\_t \\approx 0$):  
     The emotion was background noise.  
     Msg \= "Context: You met them while you were already feeling \[Mood\] that day."  
     (This phrasing helps the user dissociate the person from the negative feeling.)

## **6\. Scenario Example (Prosthetic Mode)**

**Scenario:** The user (wearer) is sundowning—confused and agitated (High negative $M\_t$). They see a neutral neighbor, Mr. Jones, who says hello.

1. **Phasic Input (**$e\_t$**):** Mr. Jones says "Hello\!" in a calm, steady voice. The system analyzes the audio features (volume, pitch) and detects a neutral/positive signal ($e\_t \\approx \\text{Joy/Neutral}$).  
2. **Accumulation (**$M\_t$**):** The internal state remains "Agitated/Fear" due to previous events (accumulated vocal tension from the wearer).  
3. **Storage:**  
   * Memory: "Mr. Jones greeting."  
   * event\_emotion: Neutral/Joy (derived from audio prosody).  
   * ambient\_mood: Agitated (derived from wearer's prior state).  
4. **Later Retrieval:** User asks, "Who is Mr. Jones?"  
5. **Grounding Output:**  
   * The system sees the mismatch: The user was Agitated ($M\_t$), but the specific interaction was Neutral ($e\_t$).  
   * **System Response:** "That is Mr. Jones. You spoke yesterday. **Note: You were feeling agitated that afternoon, but this interaction was pleasant.**"  
   * **Outcome:** The system correctly identifies Mr. Jones as safe, preventing the user from falsely attributing their agitation to him.

## **7\. Summary of Changes**

| Feature | Standard RAG | Cortext (Prosthetic Mode) |
| :---- | :---- | :---- |
| **Emotion** | Label on a chunk. | Dual vectors: **Phasic** ($e\_t$) and **Tonic** ($M\_t$). |
| **Duration** | Milliseconds. | Minutes to Days. |
| **Storage** | Content relevance. | Records both **Event** and **Background** separately. |
| **Retrieval** | Semantic only. | Semantic only \+ **Attribution-Aware Display** (distinguishes cause vs. coincidence). |
| **Knobs** | N/A | $F$ is forced to 1.0 during queries to ensure objectivity. |

