Here is the natural language overview of the algorithms, explaining **what** they do and **how** the three knobs shape their "personality."

### 1. Focus: The Lens

**Algorithm 1: Focus Priors**

* **What it does:** Sets the system's baseline "prescription" for attention. It determines how picky the system should naturally be before seeing any data.
* **The Knobs:** **Focus** sets the initial width of the lens. High Focus starts with a narrow spotlight (ignoring peripherals); Low Focus starts with a wide-angle lens (taking it all in).

**Algorithm 2: Dynamic Update per Signal (Focus)**

* **What it does:** actively adjusts the lens in real-time. If the current context is messy, it might squint (narrow) to find clarity; if the context is clear, it might relax.
* **The Knobs:** **Focus** controls how quickly the lens reacts to change. High Focus resists widening even when things are chaotic, forcing the system to "tunnel vision" on what fits.

### 2. Sensitivity: The Nervous System

**Algorithm 3: Sensitivity Priors**

* **What it does:** Sets the system's resting heart rate and emotional baseline. It determines how easily the system gets excited or bored.
* **The Knobs:** **Sensitivity** sets the gain. High Sensitivity means the system wakes up expecting to be surprised and moved; Low Sensitivity means it wakes up calm and detached.

**Algorithm 4: Dynamic Update per Signal (Sensitivity)**

* **What it does:** The adrenaline spike. It monitors the current input for emotion and novelty. If it sees something shocking, it temporarily lowers the barriers to learning, allowing that specific moment to burn in deeper.
* **The Knobs:** **Sensitivity** controls the magnitude of the spike. High Sensitivity turns a small surprise into a major learning event.

### 3. Stability: The Anchor

**Algorithm 5: Stability Priors**

* **What it does:** Sets the weight of the anchor. It determines the default lifespan of a memory and how much evidence is required to change the system's mind.
* **The Knobs:** **Stability** sets the drag. High Stability assumes memories should last a long time and that the current topic won't change abruptly.

**Algorithm 6: Dynamic Update per Signal (Stability)**

* **What it does:** Checks if the anchor is dragging. It compares how long memories *are* lasting versus how long they *should* last. If old memories are still useful, it reinforces the anchor; if they are failing, it loosens up.
* **The Knobs:** **Stability** dictates how stubborn the system is. High Stability will ignore a few missed predictions before deciding to change its fundamental retention rates.

### 4. Adaptation & Thresholds

**Algorithm 7: Metric Weight Blending**

* **What it does:** The "DJ Mixer." It has multiple input channels (relevance, novelty, surprise, drift) and fades between them to decide what "Important" looks like right now.
* **The Knobs:** The knobs preset the faders. High **Focus** boosts the "Relevance" channel. High **Sensitivity** boosts the "Novelty" channel. High **Stability** slows down how fast the DJ moves the sliders.

**Algorithm 8: Adaptive Threshold Evolution**

* **What it does:** The "Dam." It decides how high the water level needs to be before it spills over into long-term storage. It constantly raises or lowers this wall to keep the flow of new memories steady (Homeostasis).
* **The Knobs:** **Stability** builds a thicker, higher dam. **Sensitivity** installs a floodgate that can be thrown open instantly when emotional water pressure gets high.

### 5. Structural Metrics (The Dashboard)

**Algorithm 10: Coherence / Integration**

* **What it does:** Asks, *"Does what I'm seeing right now fit together?"* It measures if the current inputs tell a consistent story or a scrambled one.
* **The Knobs:** High **Focus** uses this to filter out noise; if coherence drops, it blocks inputs rather than learning the confusion.

**Algorithm 11: Contextual Entropy / Focus Spread**

* **What it does:** Asks, *"Am I distracted?"* It measures if the system is recalling one clear topic or five different unrelated things.
* **The Knobs:** High **Stability** allows for more neighbors (checking more potential links) before deciding if the spread is too wide.

**Algorithm 12: Trajectory Drift**

* **What it does:** Asks, *"Have we changed the subject?"* It tracks the conversation's path. If the topic moves too far from where it started, it declares an "Episode Boundary" (like turning the page in a book).
* **The Knobs:** **Stability** determines the page length. High Stability lets a conversation wander further before deciding it's a new chapter.

**Algorithm 13: Logprob-Derived Surprise**

* **What it does:** Asks, *"Did I expect you to say that?"* It uses the LLM's own confusion (probability) as a signal for importance.
* **The Knobs:** **Stability** acts as a skeptic. High Stability suppresses this signal, assuming the surprise is just noise/error rather than a revelation.

### 6. Learning & Feedback (The Gym)

**Algorithm 14: Memory Strength Adjustment (Base)**

* **What it does:** "Use it or lose it." Every memory decays over time unless it is recalled.
* **The Knobs:** **Sensitivity** is the muscle builder (recalling boosts strength fast). **Stability** is the muscle endurance (strength fades slower).

**Algorithm 15, 16, 17: Feedback Adjustments**

* **What it does:** The "Coach." When a memory is retrieved and actually helps the system generate a good answer, these algorithms look back and tweak the knobs for next time.
  * **Focus Feedback:** "That precise memory helped? Good, narrow the lens."
  * **Sensitivity Feedback:** "That weird fact helped? Good, look for more weird stuff."
  * **Stability Feedback:** "That old fact helped? Good, keep old stuff longer."

**Algorithm 18 & 19: Influence Updates**

* **What it does:** The "Credit Assignment." It figures out exactly *which* memory changed the output and rewards it specifically, rather than just rewarding everything in the window.
* **The Knobs:** **Focus** determines how discriminative the reward is (only the very best get a trophy).

### 7. Cognitive Dynamics (The Brain)

**Algorithm 20: Memory Reconsolidation**

* **What it does:** "Editing the Past." When you recall a memory, it becomes pliable. You might rewrite it slightly based on new context.
* **The Knobs:** **Stability** protects the past. High Stability prevents old memories from being overwritten by new biases.

**Algorithm 21: Retrieval Competition**

* **What it does:** "The Crowded Room." If ten memories scream "pick me!", this suppresses the weaker ones so the loudest can be heard clearly.
* **The Knobs:** **Focus** acts as the silencer. High Focus brutally suppresses competitors to ensure only the single best match wins.

**Algorithm 22: Predictive Pre-activation**

* **What it does:** "Anticipation." If A usually follows B, and we just saw B, this warms up A so it's ready to jump out instantly.
* **The Knobs:** **Focus** limits how far ahead we look. **Sensitivity** makes the system jumpy—it might pre-activate too much based on faint clues.

**Algorithm 23: Emotional Consolidation Tags**

* **What it does:** "Flashbulb Memories." It tags highly emotional events so they bypass normal decay rules.
* **The Knobs:** **Sensitivity** lowers the bar for what counts as "emotional," creating more permanent tags.

**Algorithm 24: Working Memory Gates**

* **What it does:** "The Workbench." Determines how many items you can hold in your head at once (Miller's Law).
* **The Knobs:** **Sensitivity** buys a bigger workbench. **Focus** demands the tools on the bench be strictly related to the task.

**Algorithm 25: Metacognitive Monitoring**

* **What it does:** "Tip of the Tongue." It knows when it *should* know something but can't find it, triggering a different search strategy.
* **The Knobs:** **Focus** makes the system arrogant; it assumes if it can't find it immediately, it doesn't exist.

**Algorithm 26: Serial Position Effects**

* **What it does:** "First and Last." It artificially boosts the importance of the first thing you saw (Primacy) and the last thing you saw (Recency).
* **The Knobs:** **Sensitivity** exaggerates this effect, making beginnings and endings feel much more significant than the middle.

**Algorithm 27: Marginal Utility Gate**

* **What it does:** "The Bouncer." (See previous explanation). Decides if an interruption is worth the distraction cost.

### 8. Consolidation (The Librarian)

**Algorithm 28: Consolidation Triggers**

* **What it does:** Decides *when* to clean up the library. It waits for the system to be idle so it doesn't interrupt active thinking.
* **The Knobs:** **Stability** makes the librarian patient; it waits for a long quiet period before starting work.

**Algorithm 29: Consolidation Scoring**

* **What it does:** Decides *what* to merge. It looks for duplicate books or chapters that can be summarized into one volume.
* **The Knobs:** **Focus** is a strict editor; it only merges things that are almost identical. **Stability** protects the classics from being summarized away.

**Algorithm 30: Graph Construction**

* **What it does:** "Connecting the Dots." It draws lines between books (memories) that mention the same characters or themes.
* **The Knobs:** **Focus** only draws lines between obvious connections. **Sensitivity** draws lines between anything that vaguely smells similar.

**Algorithm 31: Graph-Augmented Retrieval**

* **What it does:** "The Reference Desk." When you ask for a book, it also gives you the books referenced in the bibliography.
* **The Knobs:** **Focus** limits the bibliography to 1 or 2 items. **Sensitivity** dumps the whole shelf on the desk.

**Algorithm 32: Adaptive Consolidation Rate**

* **What it does:** "Work Speed." Determines how many books the librarian processes per hour.
* **The Knobs:** **Stability** sets a steady, slow pace. **Sensitivity** rushes to clean up the mess as soon as it's made.

**Algorithm 33: Graph Goal Alignment**

* **What it does:** "The Compass." It checks if the new information moves you closer to your stated goals.
* **The Knobs:** **Focus** ignores anything that isn't a direct step toward the goal. **Sensitivity** accepts "scenic routes" that might be helpful later.
