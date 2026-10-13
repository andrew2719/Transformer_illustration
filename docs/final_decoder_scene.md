I want one final extension to the existing completed demo:

DECODER DEEP DIVE

Do NOT redesign or replace any existing scenes.

The current presentation already teaches encoder self-attention deeply and
shows the full Transformer structurally.

This new section should continue naturally from the existing final
Transformer / paper-scale view and briefly go INSIDE THE DECODER.

The goal is NOT another detailed toy-model calculation.

The goal is to show what actually flows through the decoder using the
ORIGINAL PAPER'S BASE-MODEL DIMENSIONS:

d_model = 512
h = 8
d_k = d_v = 64
d_ff = 2048
N = 6 decoder layers

We will NOT display 512 individual numbers.

Represent high-dimensional vectors using compressed notation such as:

[0.45, 0.77, 0.91, ..., 1.68]

All such displayed values are ILLUSTRATIVE, not actual trained Transformer
weights or activations.

The MATRIX SHAPES and OPERATIONS, however, must match Vaswani et al. 2017.

==================================================
IMPORTANT: REUSE THE EXISTING TRANSLATION EXAMPLE
==================================================

Do NOT introduce a third unrelated English/French example.

Reuse the translation example that already exists in Scene 10.

Derive the source token count s and decoder-prefix length t directly from
the existing token arrays.

Do not hard-code a token count if punctuation/tokenization makes the
actual sequence length different.

Use notation such as:

source length = s
decoder prefix length = t

and show the concrete dimensions calculated from the existing sequence.

==================================================
PLACEMENT
==================================================

Place this AFTER the full Transformer / paper-scale reveal and BEFORE
Explore.

Prefer making this an optional guided branch or a short group of guided
beats titled:

INSIDE THE DECODER

The existing main presentation must still work without entering this
section.

Target live duration:

about 60–90 seconds

Do not turn this into another five-minute walkthrough.

==================================================
DECODER SCENE 1 — START GENERATION
==================================================

Begin from the full Transformer view.

Keep the encoder visible on the LEFT.

Dim everything except:

- encoder output
- decoder
- cross-attention connection

Then zoom into the bottom of the decoder.

Show:

<START>

as the only decoder input initially.

Animate:

token
→ embedding
→ × √512
→ positional encoding
→ decoder representation

Represent it as a compressed vector:

x_START =
[0.45, 0.77, 0.91, ..., 1.68]

Label:

shape = 1 × 512

The values are illustrative.

The dimensionality is paper-accurate.

Caption:

"At the first generation step, the decoder only has <START>."

==================================================
DECODER SCENE 2 — MASKED MULTI-HEAD SELF-ATTENTION
==================================================

Now send this 1×512 representation into:

MASKED MULTI-HEAD SELF-ATTENTION

Show that ALL 8 heads receive the entire 512-dimensional representation.

Do NOT show the 512 dimensions as eight pre-cut chunks.

Make this visually explicit:

Each head independently projects the whole 512D representation.

For Head 1 show:

W_Q : 512 × 64
W_K : 512 × 64
W_V : 512 × 64

Then:

(1×512)(512×64)
→ Q : 1×64

(1×512)(512×64)
→ K : 1×64

(1×512)(512×64)
→ V : 1×64

Show compressed vectors such as:

Q₁ = [0.72, -1.34, ..., 0.21]
K₁ = [0.15, 0.87, ..., 0.62]
V₁ = [0.62, -0.31, ..., 0.19]

Clearly label:

illustrative values

Then show Heads 2–8 appearing in parallel.

Each head should be represented as:

512
 ↓ projection
64-dimensional Q/K/V

==================================================
FIRST-STEP MASK INTUITION
==================================================

Because there is only ONE decoder token initially:

t = 1

Head 1's self-attention score matrix is:

QKᵀ:

(1×64)(64×1)
→ 1×1

Show:

[s]

Then:

s / √64
=
s / 8

Then:

softmax([s/8])
=
[1]

Visually communicate:

"There is only one available decoder position, so it receives all the
self-attention."

Do not pretend the causal mask is interesting yet.

At t = 1 there is no future decoder position to block.

Then show:

Head 1 output:
1 × V₁
→ 1×64

Do the remaining seven heads rapidly.

==================================================
COMBINE THE 8 HEADS
==================================================

Show:

Head 1 → 1×64
Head 2 → 1×64
...
Head 8 → 1×64

Physically move them together.

Concat:

8 × 64
→ 512

More precisely display:

Concat(head₁,...,head₈)
:
1 × 512

Then:

W_O:
512 × 512

so:

(1×512)(512×512)
→ 1×512

Then briefly animate:

Residual
+
LayerNorm

Keep the representation compressed:

[0.82, -0.23, 0.45, ..., 1.14]

Caption:

"Masked self-attention transformed the decoder prefix.
It has not generated a word yet."

==================================================
DECODER SCENE 3 — CROSS-ATTENTION
==================================================

THIS SHOULD BE THE MAIN PAYOFF OF THE DECODER DEEP DIVE.

Keep the decoder representation on the RIGHT.

Bring the encoder output clearly into view on the LEFT.

Suppose the encoder source length is:

s

Then show the encoder output shape:

E:
s × 512

Represent it visually as s compressed rows:

token_1  [0.31, -0.20, ..., 1.10]
token_2  [0.07,  0.90, ..., -0.40]
...
token_s  [-0.60, 0.24, ..., 0.75]

Do not imply these are real trained activations.

They are illustrative vectors showing shape.

==================================================
CROSS-ATTENTION Q / K / V SOURCES
==================================================

Visually reuse the existing color system:

CYAN = Q
AMBER = K
GREEN = V

Now explicitly animate:

DECODER STATE
      ↓
      Q

while:

ENCODER OUTPUT
      ↓
     K,V

For ONE cross-attention head show:

Decoder Q:

(1×512)(512×64)
→
Q : 1×64

Encoder K:

(s×512)(512×64)
→
K : s×64

Encoder V:

(s×512)(512×64)
→
V : s×64

This origin difference must be visually unmistakable.

Put a small caption:

"Q from decoder · K,V from encoder"

==================================================
CROSS-ATTENTION SCORES
==================================================

Now show:

QKᵀ

Shapes:

(1×64)(64×s)
→
1×s

Animate the result as one attention score for every source token.

For the existing source sequence, show an illustrative set of scores.

Then:

divide by √64 = 8

then:

softmax

produces attention weights over the SOURCE positions.

For example visually:

source token 1    0.xx
source token 2    0.xx
source token 3    0.xx
...

These values may be pedagogically chosen.

Label:

"illustrative cross-attention weights"

Do NOT imply these came from a trained translation model.

The important fact is the shape:

1 × s

==================================================
WEIGHTED ENCODER VALUES
==================================================

Use the same visual language learned earlier.

Each cross-attention weight multiplies its corresponding ENCODER Value:

α_j V_j

Weighted green Value vectors flow from the encoder into the decoder.

Then:

Σ_j α_j V_j

forms the cross-attention head output:

1 × 64

Show the other seven cross-attention heads in parallel.

Then:

8 heads × 64
→ Concat
→ 1×512
→ W_O
→ 1×512

Then:

Residual + LayerNorm

This should visually reinforce:

"Cross-attention lets the current decoder representation gather
information from the encoded source sentence."

==================================================
DECODER SCENE 4 — FEED-FORWARD + SIX LAYERS
==================================================

Now show the position-wise FFN:

1×512
→
1×2048
→ ReLU
→
1×512

Then:

Residual + LayerNorm

This completes ONE decoder layer.

Do NOT animate all six layers independently.

Instead show the completed decoder layer duplicating upward as ghosted
layers:

Layer 1
Layer 2
...
Layer 6

Display:

Paper: N = 6

Caption:

"Each decoder layer has its own learned parameters."

The encoder output should remain available to the cross-attention
sublayer of every decoder layer.

==================================================
DECODER SCENE 5 — 512 NUMBERS BECOME A TOKEN
==================================================

After Layer 6, focus on the final decoder representation at the LAST
decoder position.

Show:

h_last =
[0.72, -1.25, 0.33, ..., 0.91]

shape:

1 × 512

Important message:

"These 512 numbers are not themselves a word."

Now animate:

h_last
→ Linear / vocabulary projection
→ logits
→ Softmax

Use:

1×512
×
512×|Vocab|
→
1×|Vocab|

Do NOT hard-code a particular vocabulary size as though it were a fact
from the paper unless the project already has one.

Use:

|Vocab|

or explicitly say:

"example vocabulary size"

if showing a concrete number.

Show only a few vocabulary entries:

target_token_A     ...
target_token_B     ...
desired next token ...
...
thousands more

Then softmax.

Highlight the selected next token.

Clearly label:

"illustrative next-token probabilities"

The vocabulary projection / softmax should NOT be described as another
attention head.

==================================================
TOKEN FLIES INTO OUTPUT
==================================================

Take the chosen token and physically append it to the decoder output
sequence.

Then visually feed that token back into the decoder input.

Show:

predict
↓
append
↓
feed back
↓
predict again

This connects directly to the autoregressive loop already present in the
existing demo.

==================================================
DECODER SCENE 6 — NOW THE MASK ACTUALLY MATTERS
==================================================

This is the important second generation step.

After the first generated token is appended, the decoder prefix now has:

t = 2

positions.

Show the two compressed 512D vectors stacked:

X_decoder:

2 × 512

For each head:

Q:
2 × 64

K:
2 × 64

V:
2 × 64

Then:

QKᵀ:

(2×64)(64×2)
→
2×2

Show a small score matrix:

[ s00  s01 ]
[ s10  s11 ]

Now apply the causal mask.

The upper-right future connection becomes:

−∞

Show:

[ s00   −∞  ]
[ s10   s11 ]

Then row-wise softmax:

[ 1      0    ]
[ α      1−α  ]

The exact α may be illustrative.

The key visual point is:

FIRST position cannot look ahead.

SECOND position may look at:
- the previous position
- itself

Caption:

"Now the causal mask matters."

==================================================
SECOND-STEP CROSS-ATTENTION
==================================================

Briefly show that cross-attention now has:

Decoder Q:
2 × 64

Encoder K,V:
s × 64

So:

QKᵀ:

(2×64)(64×s)
→
2×s

Do NOT calculate every value.

Just show the matrix growing from:

1×s
to
2×s

as the decoder prefix grows.

==================================================
PREDICT THE NEXT TOKEN
==================================================

After all six decoder layers:

there are now:

2 × 512

final decoder representations.

Highlight ONLY the LAST row.

Caption:

"Generation uses the representation at the newest decoder position."

That last 512D vector goes through:

Linear
→ Softmax
→ next token

Append the selected token.

Then show the process quickly repeating:

t = 1
→ t = 2
→ t = 3
→ ...

until the end token is produced.

==================================================
FINAL DECODER SUMMARY
==================================================

Finish with a clean dimensional summary.

For decoder prefix length t
and encoder source length s:

Decoder representation:
t × 512

Masked self-attention per head:

Q,K,V:
t × 64

Scores:
t × t

8 heads:
→ t × 512

Cross-attention per head:

Q from decoder:
t × 64

K,V from encoder:
s × 64

Scores:
t × s

8 heads:
→ t × 512

FFN:
t × 512
→ t × 2048
→ t × 512

Repeat decoder layer:
6×

Final newest position:
1 × 512

Vocabulary projection:
1 × 512
→ 1 × |Vocab|

Softmax:
next-token probabilities

==================================================
FINAL INTUITION
==================================================

End with two lines:

"Masked self-attention mixes information across the output generated so far."

"Cross-attention brings information in from the encoded input."

Then return to the full Transformer shot or Explore mode.

==================================================
ACCURACY RULES
==================================================

Do NOT imply:

- 512 dimensions are simply split into eight fixed chunks.
  Each head projects the entire 512D representation into its own 64D
  Q/K/V space.

- individual heads directly vote for words.

- attention outputs are vocabulary probabilities.

- illustrative vectors/probabilities came from the original trained model.

- all six decoder layers share weights.

- the first decoder step has a meaningful future mask when only <START>
  exists.

- cross-attention uses decoder K/V.
  Q comes from decoder; K/V come from encoder.

Maintain the original 2017 post-LN structure.

==================================================
PRESENTATION SAFETY
==================================================

Make the Decoder Deep Dive skippable.

Do not disturb the existing 5–10 minute main presentation.

If I am short on time, I should be able to go directly from the existing
full Transformer / paper-scale scene into Explore.

If I have time or someone asks:

"Okay, but what exactly happens inside the decoder?"

I should be able to enter this branch and show it.

==================================================
TESTING
==================================================

After implementation verify:

- shapes are correct at every decoder stage
- √d_k = √64 = 8 in paper-scale attention
- masked future cells become exactly 0 after softmax
- cross-attention score shape is t×s
- masked self-attention score shape is t×t
- 8×64 concatenates to 512
- W_O maps 512→512
- FFN maps 512→2048→512
- final vocabulary projection uses the newest decoder position
- the section can be skipped
- entering/leaving it does not corrupt existing guided navigation
- 1920×1080 and 1366×768 both fit

Do not redesign the rest of the demo.