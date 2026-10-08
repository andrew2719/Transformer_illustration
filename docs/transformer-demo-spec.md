You are an expert frontend engineer, mathematical animator, and deep-learning educator.

I need you to build a COMPLETE, polished, interactive HTML demonstration for a live university graduate-level Deep Learning presentation on:

“Attention Is All You Need”
Vaswani et al., 2017

I am the DEMO PRESENTER, not the historical-context presenter and not the architecture/diagram presenter.

The purpose of my demo is therefore NOT to give another slide lecture.

The purpose is:

SHOW ONE CENTRAL TRANSFORMER MECHANISM ACTUALLY HAPPENING.

The audience should watch a sentence become vectors, watch Query/Key/Value vectors form, watch dot products get computed, watch scaling and softmax happen numerically, watch Value vectors get weighted and combined, and finally understand intuitively what self-attention is doing.

The experience should be suitable for a 5–10 minute live classroom presentation.

========================================================
SOURCE OF TRUTH
========================================================

If the original “Attention Is All You Need” paper is in the folder, READ IT.

Use the paper as the source of truth for:
- scaled dot-product attention
- Q, K, V
- multi-head attention
- positional encoding
- residual connection + layer norm
- feed-forward block
- encoder self-attention
- decoder masking, if shown
- architecture dimensions

Do NOT silently replace the 2017 architecture with modern Transformer conventions.

Important original-paper facts:

Attention(Q,K,V)
=
softmax(QK^T / sqrt(d_k)) V

The base model uses:
- N = 6 encoder layers
- N = 6 decoder layers
- d_model = 512
- h = 8 heads
- d_k = d_v = 64
- d_ff = 2048

However:

DO NOT use those large dimensions for the teaching calculation.

Use a deliberately tiny Transformer attention example:

d_model = 4
heads = 2
d_k = d_v = 2

Clearly label this everywhere relevant:

“Tiny teaching model — dimensions reduced so every calculation can be shown.”

Also state:

“The mechanism is real Transformer attention. The small vectors and weights are illustrative teaching values, not weights from the original trained model.”

========================================================
COURSE / DEMONSTRATION GOAL
========================================================

This is a LIVE INTERACTIVE TECHNICAL ARTIFACT.

The demonstration should be:
- accurate
- interactive
- visually intuitive
- mathematically real
- simple enough to run live
- understandable without lots of text

The audience should be able to change something and immediately see what changes.

The most important outcome is that someone who did not understand self-attention before should understand it after watching this demo.

Do not attempt to reproduce the paper’s experimental results.

Demonstrate the mechanism.

========================================================
VISUAL DIRECTION
========================================================

Create a sophisticated mathematical-explainer aesthetic inspired by high-quality animated mathematics videos.

Do NOT literally clone any particular creator’s artwork.

Desired qualities:

- near-black / very dark navy background
- large amount of negative space
- elegant mathematical typography
- smooth object transformations
- thin luminous network wires
- glowing nodes
- softly animated vectors
- curved Bézier attention connections
- geometric diagrams
- equations integrated into the animation
- deliberate and restricted color palette
- almost no decorative UI
- smooth camera-like zooms and transitions
- visual continuity between steps

The animation itself should explain the mathematics.

Do NOT make this look like:
- Bootstrap
- a SaaS dashboard
- an admin panel
- a documentation website
- a generic AI landing page
- a collection of cards
- a slide deck embedded in HTML
- a dense Jupyter notebook

I want something that feels like:

“A mathematical diagram has come alive.” - inspiration from 3Blue1Brown youtube channel.

========================================================
COLOR SEMANTICS
========================================================

Colors MUST have consistent semantic meaning.

Use approximately:

QUERY:
cyan / blue

KEY:
amber / orange

VALUE:
green

POSITIONAL ENCODING:
purple / violet

NEUTRAL REPRESENTATIONS:
white / pale blue-gray

ATTENTION STRENGTH:
brightness + line thickness

MASKED / DISABLED:
muted red

Do not randomly recolor objects between scenes.

The audience should unconsciously learn:

BLUE asks.
ORANGE matches.
GREEN carries information.

========================================================
CORE PHILOSOPHY
========================================================

Every animation must correspond to a computation.

GOOD:

A token vector moves through W_Q and becomes q.

A Query physically moves toward a Key.

Their matching components highlight.

The multiplication terms appear.

The products sum.

The resulting score moves into the score row.

The score row gets divided by sqrt(d_k).

Softmax transforms raw scores into normalized bars.

Attention weights multiply Value vectors.

Weighted Value vectors move back toward the Query token.

They combine into a new representation.

BAD:

Random particles.
Meaningless glow.
Spinning elements.
Animated backgrounds.
Decorative motion unrelated to computation.

For every animation ask:

“What mathematical operation is this motion explaining?”

========================================================
DEFAULT SENTENCE
========================================================

Use this guided example:

“The animal didn't cross the street because it was tired.”

Use simple whitespace/punctuation tokenization for the teaching demo.

Do NOT spend time explaining production tokenizers or BPE.

Display something like:

[The]
[animal]
[didn't]
[cross]
[the]
[street]
[because]
[it]
[was]
[tired]
[.]

The focus token should initially be:

“it”

because it gives a useful visual example of long-distance relationships.

However:

DO NOT claim that a manually constructed teaching head has “learned pronoun resolution.”

If the toy weights produce an interpretable pattern, describe it only as:

“Illustrative attention behavior.”

========================================================
PAGE STRUCTURE
========================================================

Build ONE full-screen visual stage.

It should NOT scroll during the guided presentation.

Main regions:

TOP LEFT:
small title:
ATTENTION IS ALL YOU NEED

TOP RIGHT:
small status such as:
4 / 9 · Dot Products

CENTER:
the animated mathematical visualization

BOTTOM CENTER:
Previous
Play/Pause
Next

BOTTOM:
very subtle progress timeline

Example:

Sentence
Vectors
Position
QKV
Scores
Softmax
Values
Heads
Explore

Keyboard:

Left Arrow = previous
Right Arrow = next
Space = play/pause
R = restart
E = enter Explore Mode

========================================================
GUIDED PRESENTATION
========================================================

The main presentation should consist of approximately NINE scenes.

Do not make 20 scenes.

The whole guided section should fit naturally into a 5–8 minute explanation.

Transitions should preserve object identity.

Do not destroy the screen and recreate everything every time Next is clicked.

The same objects should move, split, transform and recombine.

========================================================
SCENE 1 — THE SENTENCE
========================================================

Begin almost completely dark.

Fade in:

ATTENTION IS ALL YOU NEED

Then underneath:

“What happens to one token inside self-attention?”

Then show the sentence:

The animal didn't cross the street because it was tired.

Words appear sequentially.

Then subtle thin lines briefly connect every token to every other token.

Display only one short sentence:

“Self-attention lets every position directly inspect every other position.”

Then remove that sentence.

Keep the tokens.

No paragraph.

========================================================
SCENE 2 — TOKENS BECOME VECTORS
========================================================

Each word becomes a small token node.

Then animate each token node expanding downward into a 4-dimensional vertical vector.

For example:

        it

      ┌      ┐
      │ 0.80 │
      │-0.20 │
      │ 0.50 │
      │ 0.10 │
      └      ┘

The actual values should come from JavaScript data.

DO NOT hard-code the rendered text independently of the numerical state.

Represent every token as a tiny 4-dimensional vector.

Add a tiny annotation:

Paper:
d_model = 512

Demo:
d_model = 4

Say visually:

“Words are now vectors.”

That is enough.

========================================================
SCENE 3 — POSITION
========================================================

The vectors move slightly downward.

A purple sinusoidal wave appears underneath them.

For each token, a small purple positional vector rises toward the token embedding.

Show:

embedding
+
position
=
input representation

For ONE selected token, display the actual component-wise addition.

For example:

Embedding

[ 0.80
 -0.20
  0.50
  0.10 ]

+

Position

[ 0.14
  0.99
  0.01
  1.00 ]

=

[ 0.94
  0.79
  0.51
  1.10 ]

Use the actual numbers produced by the positional-encoding code.

Implement the sinusoidal positional encoding formula from the paper.

Add a small expandable Math button showing:

PE(pos,2i)
=
sin(pos / 10000^(2i/d_model))

PE(pos,2i+1)
=
cos(pos / 10000^(2i/d_model))

Do not leave the equations permanently dominating the screen.

========================================================
SCENE 4 — Q, K, V
========================================================

THIS IS A MAJOR VISUAL MOMENT.

Fade most tokens slightly.

Highlight:

“it”

Lift its current representation into the center.

The vector then splits smoothly into THREE branches.

LEFT:
Q
cyan

CENTER:
K
amber

RIGHT:
V
green

Animate the corresponding projection matrices appearing in front of them:

W_Q
W_K
W_V

Then show:

x_it W_Q = q_it
x_it W_K = k_it
x_it W_V = v_it

ACTUALLY CALCULATE THESE IN JAVASCRIPT.

Use 4×2 projection matrices.

For one projection, animate the matrix-vector computation in detail.

Example visual idea:

[x1 x2 x3 x4]

×

┌           ┐
│ w11  w12 │
│ w21  w22 │
│ w31  w32 │
│ w41  w42 │
└           ┘

→

[q1 q2]

Highlight matching multiplication pairs.

Then show the resulting two-dimensional Query.

Do not fully animate all three matrix multiplications slowly.

Show Query slowly.

Then Key and Value can happen quickly.

Add short intuition labels:

Q
“What am I looking for?”

K
“What do I match?”

V
“What information can I contribute?”

These are PEDAGOGICAL ANALOGIES.

They must be visually labeled as:

“intuition”

========================================================
SCENE 5 — QUERY MEETS KEYS
========================================================

THIS SHOULD BE THE MOST MEMORABLE PART.

Put the selected token “it” and its cyan Query vector on the left.

Arrange every token’s Key vector along an arc or row on the right.

Draw no connections initially.

Then animate the Query examining one Key at a time.

When Query reaches a Key:

- corresponding dimensions glow
- show component multiplication
- show summation
- produce a scalar score

Example:

q_it = [1.10, 0.40]

k_animal = [0.80, 0.60]

q · k

=
(1.10 × 0.80)
+
(0.40 × 0.60)

=
0.88 + 0.24

=
1.12

That resulting:

1.12

should physically move into a score row beside “animal”.

Then calculate the other token scores faster:

The       0.xx
animal    1.xx
didn't   -0.xx
cross     ...
...

DO NOT hard-code the score values separately.

All scores must come from:

dot(query, key)

using the same underlying vectors shown on screen.

The strongest scores should create brighter/thicker temporary wires.

========================================================
SCENE 6 — SCALE + SOFTMAX
========================================================

Move the raw scores into a horizontal mathematical pipeline.

Show:

QKᵀ

then:

QKᵀ / √d_k

Animate the denominator:

√2

because the teaching head has:

d_k = 2

The raw numerical scores should visibly change.

Example:

1.12
→
1.12 / √2
→
0.79

Do it for all scores simultaneously.

Then reveal:

softmax

The scores should visually transform into normalized attention probabilities.

Use animated horizontal bars.

Example:

animal
██████████████
0.38

tired
█████████
0.24

street
████
0.11

...

The displayed numbers must be calculated with actual JavaScript softmax.

At the bottom:

Σ attention = 1.000

The equation should assemble visually:

Attention(Q,K,V)

=

softmax(
QKᵀ / √d_k
)V

Do not merely fade the completed equation in.

Build it from the computation the audience just watched.

QKᵀ should appear first.

Then /√d_k.

Then softmax.

Leave V dimmed for the next scene.

========================================================
SCENE 7 — VALUES MOVE
========================================================

Now make V glow green.

Each token has a green Value vector.

Place its attention probability beside it.

Show:

α_j × V_j

for every token.

For one example, show actual multiplication:

0.38 × [0.50, 1.20]

=

[0.19, 0.46]

Then animate the weighted Value vectors physically flowing toward:

“it”

Use attention probability to control:

- path thickness
- brightness
- animation speed or emphasis

High weight:
bright thick stream

Low weight:
faint thin stream

As each weighted Value reaches “it”, accumulate it.

Show:

z_it
=
Σ_j α_j V_j

The final two-dimensional attention output vector should visibly form.

Then place it back underneath “it”.

Display one short line:

“‘it’ now contains information gathered from other positions.”

No longer explanation.

========================================================
SCENE 8 — MULTI-HEAD ATTENTION
========================================================

Zoom outward.

Duplicate the attention process into two parallel horizontal lanes:

HEAD 1
HEAD 2

Head 1:
cyan-orange-green palette with one slight accent

Head 2:
same semantic colors, but differentiate using labels and geometry rather than completely new colors.

Each head must have its own:

W_Q
W_K
W_V

and therefore genuinely calculate a different attention pattern.

Do not simply fake different line positions.

Show the focus token’s attention lines for both heads.

Then both output vectors travel toward the center:

head_1
head_2

→

CONCAT

Show the concatenated 4D vector.

Then:

× W_O

→

Multi-head output

ACTUALLY calculate concatenation and output projection.

Display the formula:

MultiHead(Q,K,V)

=
Concat(head_1,...,head_h)W_O

and underneath:

head_i
=
Attention(QW_i^Q, KW_i^K, VW_i^V)

Also display:

Original paper:
8 heads

Teaching demo:
2 heads

Do NOT claim:

“Head 1 is pronoun resolution”

unless using genuine trained-model evidence.

Instead label:

“Different projections → different attention patterns.”

========================================================
SCENE 9 — ZOOM OUT TO THE TRANSFORMER BLOCK
========================================================

Now zoom out dramatically.

All of the detailed Q/K/V machinery should shrink smoothly into a box labeled:

MULTI-HEAD
SELF-ATTENTION

Do NOT redraw a completely unrelated architecture.

Use the object the audience already saw and spatially collapse it.

Reveal:

input
↓
Multi-Head Attention
↓
Add + Norm
↓
Feed Forward
↓
Add + Norm

Show a residual wire as a beautiful thin loop:

input representation
───────────────────────┐
        ↓               │
   ATTENTION            │
        ↓               │
        + ←─────────────┘
        ↓
     LayerNorm

Then show the FFN very briefly as:

representation
→ 4
→ 8
→ ReLU
→ 4

and annotate:

Original paper:
512 → 2048 → 512

Demo:
4 → 8 → 4

Do NOT spend long on this.

The Diagrammer explains the complete architecture.

My demo is about understanding attention.

Finally duplicate the encoder block upward faintly:

Layer 1
Layer 2
...
Layer 6

Very briefly show that the original encoder stacks six of these.

========================================================
EXPLORE MODE
========================================================

After Scene 9, transition into:

EXPLORE IT

This mode should let the class manipulate the demonstration.

Keep controls MINIMAL.

Required interactions:

1. CLICK ANY TOKEN

When a token is clicked:
- it becomes the Query token
- its Q vector appears
- attention is recomputed
- attention curves animate outward
- probability bars update
- weighted Value flow updates

2. HEAD SELECTOR

Head 1
Head 2

Switching head must use genuinely different projection matrices and recompute everything.

3. POSITION TOGGLE

Position:
ON / OFF

When turned OFF:

Remove positional encoding from the token representations.

Recompute:

Q
K
V
scores
softmax
weighted sums

Animate the change.

Do not claim that the structure must always “collapse” in this tiny teaching network.

Instead say:

“Without positional information, the mechanism no longer receives explicit token-order information.”

4. OPTIONAL SCALE EXPERIMENT

Include one small control:

Attention scale

Options:

No scaling
Paper scaling: √d_k
Custom slider

When using custom values, visibly label:

“What-if experiment”

The default must always be:

√d_k

Changing it should immediately update:
- scaled logits
- softmax
- attention bars
- line thickness

This is a pedagogical experiment.

Do not imply that √d_k is normally a user-adjustable inference parameter.

========================================================
OPTIONAL SENTENCE EDITOR
========================================================

Allow the presenter to click:

Change sentence

Open a minimal text field.

The viewer may type a short sentence.

For arbitrary sentences:

- simple tokenization is fine
- generate deterministic pseudo-embeddings from the token text
- apply the SAME real attention math
- clearly display:

“Interactive toy Transformer — embeddings/weights are illustrative.”

Do NOT pretend arbitrary typed sentences are being processed by a pretrained Transformer unless you ACTUALLY include a pretrained model.

If implementing a genuine small pretrained model entirely in-browser is practical and robust, you may optionally include a separate:

REAL MODEL

mode.

However:

DO NOT sacrifice reliability, clarity, or the guided mathematical demo to add it.

The guided numerical demo is the priority.

========================================================
NUMERICAL ENGINE
========================================================

THE MATH MUST ACTUALLY RUN.

Implement reusable JavaScript functions:

dot(a,b)

vectorAdd(a,b)

vectorScale(a,s)

matVecMul(vector,matrix)

matMul(A,B)

transpose(A)

softmax(values)

relu(vector)

concat(...vectors)

layerNorm(vector)

positionalEncoding(position, dModel)

scaledDotProductAttention(Q,K,V,dK)

multiHeadAttention(X)

Use full floating-point precision internally.

Display values rounded to:
2 decimals normally
3 decimals where useful

NEVER use one set of numbers for the animation and another for the internal calculation.

There should be one source of truth.

========================================================
DETERMINISTIC TEACHING VALUES
========================================================

Use deterministic data.

No Math.random() for the main presentation.

Refreshing the browser must reproduce exactly the same calculation.

Choose the tiny embeddings and matrices deliberately so that:

- values are numerically readable
- softmax is not completely saturated
- different heads produce visibly different distributions
- attention probabilities are not all equal
- no values become absurdly large
- the focus token produces an interesting pattern

You may choose the exact values yourself.

BUT:

All vectors/matrices shown visually must be the exact values used by the code.

Add a developer comment near the data:

“Pedagogical toy parameters chosen for interpretability; not learned weights.”

========================================================
MATRIX VISUALIZATION
========================================================

Include an optional small button:

Attention Matrix

Clicking it should transform the token layout smoothly into an attention heatmap.

Rows:
Queries

Columns:
Keys

Use luminance to represent attention strength.

Hover / click a cell:

- highlight query token
- highlight key token
- show exact probability
- draw the corresponding curved connection

Do not make the matrix the default presentation view.

It is an optional exploration view.

========================================================
ANIMATION DETAILS
========================================================

Animations should be smooth and deliberate.

Use:

SVG
CSS transforms
requestAnimationFrame
Web Animations API

Avoid large UI frameworks unless absolutely necessary.

For attention curves:

Use SVG quadratic/cubic Bézier paths.

Animation:

stroke-dasharray
stroke-dashoffset

or animated particles moving along the path.

For weighted values:

Animate small green vector capsules traveling along the attention path.

For matrix multiplication:

Highlight:
row
column
products
sum

in sequence.

For equation transformation:

Use shared-element motion where possible.

Example:

the numerical value “1.12”

should MOVE from the dot-product calculation into the score list.

Do not independently create another “1.12” somewhere else if visual continuity is possible.

========================================================
OBJECT CONSTANCY
========================================================

This is extremely important.

The same mathematical object should feel like the same object across scenes.

Examples:

The token “it” from Scene 1
should be the same DOM/SVG object that becomes selected later.

Its vector should move into the Q/K/V projection.

Its Query should travel toward Keys.

The resulting score should move into softmax.

The probability should become the weight controlling a Value vector.

That weighted Value should move back into the token.

The audience should feel:

“I watched the data move.”

not:

“The website showed nine unrelated diagrams.”

========================================================
TEXT PHILOSOPHY
========================================================

Use VERY LITTLE prose.

Never show a paragraph when one sentence will do.

Examples of acceptable on-screen text:

“Words become vectors.”

“Position is added.”

“One vector becomes Q, K and V.”

“Compare the Query with every Key.”

“Scale.”

“Normalize.”

“Use those weights to mix Values.”

“Different projections create different heads.”

That is the preferred communication style.

The presenter will speak.

The screen should visualize.

========================================================
FORMULA DISPLAY
========================================================

Formulas should be large, beautiful and animated.

Use proper mathematical typesetting.

If external libraries are avoided, use Unicode / HTML / SVG text carefully.

If MathJax is used, ensure it loads reliably.

Required formulas:

q = xW_Q

k = xW_K

v = xW_V

score(q,k_j)
=
q · k_j

Attention(Q,K,V)
=
softmax(QK^T / √d_k)V

z_i
=
Σ_j α_ij v_j

MultiHead(Q,K,V)
=
Concat(head_1,...,head_h)W_O

FFN(x)
=
max(0,xW_1+b_1)W_2+b_2

LayerNorm(x + Sublayer(x))

Not all formulas should be visible simultaneously.

Reveal them exactly when relevant.

========================================================
NETWORK / WIRE VISUALIZATION
========================================================

I specifically want the Transformer to feel like an actual computational network.

Use simple visual motifs:

- circular or rectangular vector nodes
- thin wires
- branches
- residual loops
- parallel attention head lanes
- signals moving through wires
- multiplication blocks
- summation junctions

For Q/K/V:

              ┌── W_Q ──→ Q
x ────────────┼── W_K ──→ K
              └── W_V ──→ V

But make this geometric and animated, not ASCII.

For residual:

x ────────────────────────┐
│                         │
↓                         │
ATTENTION                  │
↓                         │
+ ←───────────────────────┘

Animate the bypass path.

For multiple heads:

input
├──── Head 1 ─────┐
└──── Head 2 ─────┤
                  ↓
               CONCAT
                  ↓
                  W_O

Signals should visually flow through these paths.

========================================================
PRESENTATION CONTROLS
========================================================

Visible controls should be very subtle.

Required:

←
Play/Pause
→

Reset

Explore

Fullscreen

No giant toolbar.

Keyboard:

ArrowLeft
ArrowRight
Space
R
F
E

When in presentation fullscreen mode:

- hide cursor after inactivity
- dim controls
- keep current step indicator visible
- no scrollbar

========================================================
PLAY MODE
========================================================

Next should advance to the next meaningful animation beat.

Play should automatically advance through the current scene’s internal animation.

The presenter should be able to:

Next
talk
Next
talk
Next

without needing precise mouse interactions.

Do not create tiny click targets.

========================================================
MATH WALKTHROUGH BUTTON
========================================================

Add:

SHOW CALCULATION

When enabled:

the current numerical operation slows down and exposes every arithmetic step.

For example:

1.10 × 0.80 = 0.88

0.40 × 0.60 = 0.24

0.88 + 0.24 = 1.12

When disabled:

the animation happens much faster.

This is useful if someone in class asks:

“Where did that number come from?”

========================================================
OPTIONAL DECODER EXPLANATION
========================================================

Do NOT make decoder mechanics part of the mandatory guided flow.

Instead include a small optional button:

“Decoder mask”

If opened:

show a miniature token-by-token decoder attention matrix.

Future positions are covered.

Display:

−∞

for masked logits.

Then softmax turns them into:

0

Visually demonstrate:

future token
→ masked
→ probability 0

Include one sentence:

“The decoder cannot attend to future output positions.”

This is OPTIONAL and should take under 30 seconds.

========================================================
ACCURACY GUARDRAILS
========================================================

Do NOT teach any of these inaccurate ideas:

- Query is literally an English-language question.
- Key is literally a database key.
- Value directly contains a word’s semantic meaning.
- Attention probabilities are guaranteed explanations of model reasoning.
- Individual heads always have clean human-interpretable roles.
- Turning positional encoding off necessarily gives one particular behavior.
- The illustrative matrices are trained weights.
- The tiny demo reproduces the paper’s translation model.
- A Transformer outputs the whole autoregressive decoder sequence simultaneously.

Use clear distinctions between:

MATHEMATICAL FACT

and

PEDAGOGICAL INTUITION

when necessary.

========================================================
ABOUT / DISCLOSURE PANEL
========================================================

Include a tiny unobtrusive:

About this demo

panel.

It should say:

“Built with LLM coding assistance.”

And:

“The demo uses a scaled-down Transformer attention calculation with pedagogical parameters. The mathematical operations follow Vaswani et al. (2017).”

Also provide editable placeholder text:

“What we asked the coding assistant to build:
[EDIT ME]

What we manually checked or fixed:
[EDIT ME]”

This is needed for course disclosure.

========================================================
TECHNICAL REQUIREMENTS
========================================================

Prefer ONE self-contained:

transformer_attention_demo.html

It should contain:

HTML
CSS
JavaScript

inside one file.

No backend.

No API key.

No server required.

Opening the file directly in Chrome should work.

Prefer zero external dependencies.

If an external dependency is absolutely necessary, minimize it and explain why.

Must work on a typical laptop connected to a classroom projector.

Target:

16:9

Primary testing size:

1920×1080

Also support:

1366×768

Responsive down to typical laptop dimensions.

========================================================
PERFORMANCE
========================================================

Target smooth animation.

Avoid expensive constant DOM rebuilding.

Prefer transforms and opacity.

Avoid animating layout-heavy properties unnecessarily.

Use requestAnimationFrame where appropriate.

Stop animation loops when not needed.

========================================================
ACCESSIBILITY
========================================================

Support:

prefers-reduced-motion

Keyboard navigation.

Visible focus states.

Sufficient contrast.

Buttons with accessible labels.

Do not rely exclusively on color to distinguish Q/K/V:
also use labels and shapes where possible.

========================================================
CODE ORGANIZATION
========================================================

Keep the code well structured.

Suggested sections:

CONFIG

DEMO DATA

MATH UTILITIES

TRANSFORMER CALCULATIONS

STATE

SVG HELPERS

TOKEN RENDERER

VECTOR RENDERER

MATRIX MULTIPLICATION ANIMATION

ATTENTION CONNECTION RENDERER

EQUATION RENDERER

SCENE 1

SCENE 2

...

EXPLORE MODE

EVENT HANDLERS

KEYBOARD CONTROLS

BOOTSTRAP / INIT

Add comments around mathematically important operations.

========================================================
IMPORTANT IMPLEMENTATION PRINCIPLE
========================================================

DO NOT create static visual values and separately calculate hidden values.

The visualization itself must render directly from the computed model state.

For example:

const scores = keys.map(k => dot(query, k));

const scaledScores =
scores.map(s => s / Math.sqrt(dK));

const weights =
softmax(scaledScores);

The bars, numbers, curves and weighted Value animations must use:

weights

from that computation.

========================================================
DEBUG / VERIFICATION MODE
========================================================

Add a hidden debug mode toggled with:

D

It can show:

- full X matrix
- positional matrix
- Q
- K
- V
- QK^T
- scaled logits
- softmax attention
- attention output
- head outputs
- concatenated output

This is for ME, the presenter, to verify the demo.

It should not appear during normal presentation.

========================================================
VISUAL QUALITY CHECK
========================================================

Before finishing, inspect the experience conceptually and ask:

Does the page look like an interactive mathematical animation?

Or does it look like a software dashboard?

If it looks like a dashboard, redesign it.

Does Next cause mathematical objects to transform?

Or merely cause new cards to appear?

If new cards simply appear, redesign it.

Can a viewer visually trace:

token
→ vector
→ query
→ dot product
→ attention score
→ softmax probability
→ weighted value
→ new representation

without reading paragraphs?

If not, redesign it.

========================================================
PRESENTATION SCRIPT SUPPORT
========================================================

Include a tiny Presenter Notes mode toggled by:

N

The notes should NOT be visible to the audience in normal mode.

For every scene include approximately one or two short presenter cues.

Example:

Scene Q/K/V:

“Each representation is projected three different ways.
The math does not assign English meanings to Q, K and V, but a useful intuition is that the Query determines what this position is looking for, the Key determines how it matches, and the Value is the information that can be mixed into the output.”

Scene Scores:

“Now nothing magical happens. We take one dot product with every Key.”

Scene Softmax:

“These are just raw compatibility scores. Scale them, then softmax converts them into positive weights that sum to one.”

Scene Values:

“This is the actual information movement. The scores decide how much of every Value reaches this position.”

Keep these short.

========================================================
FINAL EXPERIENCE
========================================================

The audience should leave understanding this exact chain:

TOKEN

↓

REPRESENTATION

↓

Q, K, V

↓

QUERY ⋅ KEYS

↓

SCORES

↓

DIVIDE BY √d_k

↓

SOFTMAX

↓

ATTENTION WEIGHTS

↓

WEIGHTED VALUES

↓

SUM

↓

NEW REPRESENTATION

↓

MULTIPLE HEADS DO THIS IN PARALLEL

That is the central story.

========================================================
WHAT I CARE ABOUT MOST
========================================================

Priority order:

1. MATHEMATICAL CORRECTNESS
2. VISUAL INTUITION
3. ANIMATION QUALITY
4. LIVE PRESENTATION RELIABILITY
5. INTERACTIVITY
6. AESTHETIC POLISH

Do not sacrifice correctness for visual effects.

Do not sacrifice explanation for excessive features.

========================================================
OUTPUT
========================================================

Return the COMPLETE IMPLEMENTATION.

Do not give pseudocode.

Do not omit CSS.

Do not omit JavaScript.

Do not say:

“you can implement...”

Actually implement it.

Do not leave TODOs except for the explicitly editable LLM-disclosure text.

Your response should contain:

1. A very short summary of what you built.
2. ONE complete HTML code block.
3. Very short instructions:
   - save as transformer_attention_demo.html
   - open in Chrome
   - keyboard controls

Before returning the code, internally verify:

- all matrix dimensions match
- dot products match displayed numbers
- softmax sums to approximately 1
- Q/K/V values displayed are the actual calculated values
- multi-head concatenation dimensions match
- all Next/Previous states work
- Reset works
- Position toggle recomputes calculations
- attention-head selection recomputes calculations
- no undefined/NaN values appear
- the file runs without a backend

Build the demo now.