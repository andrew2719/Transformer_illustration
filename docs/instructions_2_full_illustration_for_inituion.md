<!-- Add-on to docs/transformer-demo-spec.md — paper-scale epilogue (requested 2026-10-08). -->

I want one final addition to the existing demo.

Do NOT redesign the current presentation.
Do NOT replace the toy 4D calculation walkthrough.
Do NOT add a new giant feature.

This is a SMALL EPILOGUE / INTUITION ADD-ON only.

The purpose is:

After the audience understands the tiny teaching model, briefly show how the SAME operations look in the original paper-scale Transformer.

IMPORTANT:
We are NOT going to show 512 explicit numbers.
We are NOT going to do another full arithmetic walkthrough.
We are NOT pretending the 4D vectors are the actual 512D vectors.

Instead, we will show compressed vector notation like:

[0.23, -0.41, ..., 0.78]

and matrix size labels.

This should communicate:

“The same mechanism we just learned scales up to the real dimensions in the paper.”

==================================================
HIGH-LEVEL GOAL
==================================================

Add a brief final guided segment, or a brief post-Scene-10 epilogue,
before Explore mode.

It should take about 20–40 seconds live.

It should visually connect:

TINY TEACHING MODEL
→
PAPER-SCALE BASE TRANSFORMER

without becoming cluttered.

==================================================
WHAT THIS ADDITION SHOULD TEACH
==================================================

The audience should leave with this intuition:

- In the demo, we used tiny vectors so every calculation could be followed.
- In the original base Transformer, the SAME operations happen on much larger vectors.
- Tokens are represented by 512-dimensional vectors.
- Each attention head uses its own Q/K/V projections.
- In the base model there are 8 heads.
- Each head uses d_k = d_v = 64.
- The 8 head outputs are concatenated back into 512 dimensions.
- The same idea also appears in the decoder and encoder-decoder attention.

This is a SCALE intuition scene, not a second math lesson.

==================================================
ADD THIS AS A SHORT FINAL SCENE
==================================================

Add one short scene before Explore mode.

You may label it something like:

“Paper Scale”
or
“Same Mechanism, Larger Space”
or
“From Toy Model to Transformer Scale”

Do not make it text-heavy.

==================================================
SCENE A — TOY MODEL VS PAPER SCALE
==================================================

Start from the final full-Transformer shot or the encoder-attention focus.

Show a clean comparison:

Demo:
- tokens: 8
- d_model = 4
- heads = 2
- d_k = d_v = 2

Paper:
- tokens: same sequence length idea
- d_model = 512
- heads = 8
- d_k = d_v = 64

Visually show a token representation changing from:

small 4D vector

[0.94
 0.79
 0.51
 1.10]

to a compressed 512D representation like:

[0.23, -0.41, 0.05, ..., 0.78]

Make the visual transformation elegant:
the tiny explicit vector stretches into a tall/glowing compressed vector strip.

Add a caption like:

“Same computation — larger vectors.”

==================================================
SCENE B — PAPER-SCALE INPUT MATRIX
==================================================

Show the whole sentence as a matrix.

For the guided sentence, show a compact label such as:

X ∈ R^(8×512)

Use the actual current token count from the guided sentence if practical.
If the token count is 8, then yes, show 8×512.

Visually represent this as:
- 8 token rows
- each row shown as a compressed vector like:
  [0.18, -0.07, ..., 0.66]

Do NOT try to render 512 cells.

It should look like:
a stack of compressed vector rows with dimension labels.

Maybe show:
8 tokens
×
512 features

with subtle bracket/matrix notation.

==================================================
SCENE C — Q, K, V PROJECTIONS AT PAPER SCALE
==================================================

Show the same matrix X splitting into Q, K, and V projections.

Label:

W_Q ∈ R^(512×64)
W_K ∈ R^(512×64)
W_V ∈ R^(512×64)

Then show:

Q, K, V ∈ R^(8×64)

for one head.

Visually:
- X is a compressed 8×512 matrix
- it flows through W_Q / W_K / W_V
- outputs three compressed 8×64 matrices

Show one row example as:

[0.23, -0.41, ..., 0.78]
× W_Q
→
[0.11, 0.54, ..., -0.07]

Again:
do NOT do detailed arithmetic here.
This is structural intuition only.

==================================================
SCENE D — 8 HEADS IN PARALLEL
==================================================

Now show the key scale jump:

Instead of 2 heads, show 8 heads.

Display:

Head 1
Head 2
...
Head 8

Each head should be visually represented as a smaller compressed lane.

For each head, label:

Q_h, K_h, V_h : 8×64

or simply:
“8×64 per head”

Show the heads in parallel.

Then animate:

Head outputs
→ Concat
→ 8×512
→ W_O
→ 8×512

Label:

Concat(head_1, ..., head_8) ∈ R^(8×512)

W_O ∈ R^(512×512)

This should feel like:
“same thing as before, but wider and more parallel.”

==================================================
SCENE E — WHERE THIS SITS IN THE FULL TRANSFORMER
==================================================

Tie this scale intuition back into the full architecture.

Briefly highlight:

- encoder self-attention
- decoder masked self-attention
- encoder-decoder attention

Do NOT do another long walkthrough.

Just show that the same paper-scale logic appears in:
- encoder self-attention
- decoder masked self-attention
- cross-attention

For cross-attention, you can briefly label:

Decoder Q:  t×64 per head
Encoder K,V: s×64 per head

or a similarly simple notation.

Again:
no detailed numbers needed.

The message is:

“The same attention mechanism appears throughout the architecture.”

==================================================
FINAL CAPTION
==================================================

End this paper-scale epilogue with a short line like:

“We used tiny dimensions to make every step visible.
The original Transformer uses the same operations at much larger scale.”

or

“The math did not change.
Only the dimensionality and number of heads increased.”

This should then transition into Explore mode.

==================================================
VISUAL STYLE
==================================================

Preserve the existing dark mathematical-explainer style.

Use the same color semantics:
- cyan = Q
- amber = K
- green = V
- purple = positional encoding

The compressed paper-scale vectors should look elegant:
- glowing strips
- compressed numeric notation with ellipses
- matrix brackets
- dimensional labels
- subtle morphing from toy vectors to compressed higher-dimensional vectors

Avoid dashboard panels.
Avoid long text.
Avoid tables that feel like spreadsheets.

==================================================
ACCURACY GUARDRAILS
==================================================

Very important:

- Do NOT imply that the 4D toy vectors are literally truncated versions of the real 512D vectors.
- Do NOT imply that the paper-scale compressed values are actual trained values from the Vaswani model.
- Clearly frame them as illustrative compressed representations.
- Do NOT add a second full numeric walkthrough.
- Do NOT clutter the final demo.

This scene is for intuition only.

==================================================
IMPLEMENTATION PREFERENCE
==================================================

Implement this as a brief guided scene before Explore mode.

If adding a full new scene is awkward, it may be a compact multi-beat
extension of the final Transformer scene.

Best outcome:
- small
- elegant
- visually impressive
- mathematically honest
- clearly tied to the paper dimensions

==================================================
DELIVERABLE
==================================================

Add this feature carefully without destabilizing the current demo.

Then test:
- forward/back navigation into and out of this new scene
- fullscreen layout
- 1920×1080
- 1366×768
- no overlap with existing final Transformer scene
- transition into Explore still works

Keep the current demo intact.
