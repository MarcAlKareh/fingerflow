# FingerFlow fingering model

This document describes the mathematics behind `backend/engine`. It is written for the two of us, so it states the model exactly rather than motivating every choice at length. References are at the end.

## 1. Problem statement

A hand is given a sequence of notes with MIDI pitches, onset times and durations in seconds. Notes struck at the same instant form an *event*. Let the events be $E_1, \dots, E_K$ with onsets $t_1 < \dots < t_K$. At event $k$ the *sounding set* $S_k$ is the set of notes struck at $E_k$ together with every earlier note that is still held at $t_k$.

A *state* $s_k$ is an assignment of fingers $\{1,\dots,5\}$ to $S_k$ that is injective and monotone in pitch: for the right hand a higher key gets a higher finger number, for the left hand a lower key gets a higher finger number. With $|S_k| = m$ there are $\binom{5}{m}$ states, at most 10. A held note keeps the finger that struck it, which is a hard constraint between consecutive states.

The fingering is the state path $s_1, \dots, s_K$ minimising

$$
C(s_{1:K}) = \sum_{k=1}^{K} \Big[ w_s \cdot \phi_s(k, s_k) + w_t \cdot \phi_t(k, s_{k-1}, s_k) + w_2 \cdot \phi_2(k, s_{k-2}, s_{k-1}, s_k) \Big]
$$

where the $\phi$ are non-negative feature vectors described in Section 4 and the $w$ are weight vectors. The cost is linear in the weights, which is what makes them learnable (Section 6).

## 2. Keyboard geometry

Distances are physical. On a standard keyboard the octave spans 164.5 mm, a white key is 23.5 mm wide at the front and a black key 13.7 mm. At the back of the key bed the twelve keys of an octave occupy nearly uniform slots, so the rear coordinate of a key with MIDI number $n$ is

$$
x_{\text{rear}}(n) = (n + \tfrac12)\,\frac{164.5}{12}\ \text{mm},
$$

while a white key is normally played at its front, $x_{\text{front}}(n) = (7\,\text{oct}(n) + w(n) + \tfrac12)\cdot 23.5$ mm with $w$ the white-key index. The lateral distance between two keys uses front coordinates when both are white and rear coordinates otherwise, because a hand that has any finger on a black key moves in and plays its white keys at the rear heads.

The consequence is that a "semitone" is not one distance: E to F is 23.5 mm, C to C# is 13.7 mm, and a white-key third is 47 mm whether major or minor. Narrow keyboards (DS6.0 with a 152 mm octave, DS5.5 with 140 mm) are supported by changing one constant.

## 3. Hand model

For each finger pair $(a, b)$ with $a < b$ Parncutt et al. (1997) give six signed bounds in semitones,

$$
\text{MinPrac} \le \text{MinComf} \le \text{MinRel} \le \text{MaxRel} \le \text{MaxComf} \le \text{MaxPrac},
$$

the relaxed, comfortable and practical ranges of the span. The table in `hand.py` follows Parncutt with two calibrations: the crossing side of the thumb pairs is widened so that a thumb-under across a fourth (the C major arpeggio) is comfortable, and the outward comfortable and practical bounds of the non-thumb pairs are one semitone wider because our spans are physical white-key distances rather than semitone counts.

The bounds are converted to millimetres and scaled by the player's hand:

$$
B_{\text{mm}} = B_{\text{st}} \cdot \frac{164.5}{12} \cdot \sigma, \qquad \sigma = \operatorname{clip}\!\Big(\frac{\text{span}_{\text{cm}}}{21.0},\ 0.6,\ 1.45\Big),
$$

where span is the maximum spread from little finger to thumb. The reference 21.0 cm is the hand that Parncutt's table describes: its MaxPrac(1,5) of 15 semitones is a minor tenth, about 205 mm between key centres. A 23 cm hand therefore gets $\sigma = 1.10$ and can take a major tenth (211.5 mm) at practical cost; an 18 cm hand cannot.

The *natural span* of a pair is signed. Let $f_{\text{lo}} < f_{\text{hi}}$ be the two finger numbers and $x_{\text{lo}}, x_{\text{hi}}$ the keys they play. For the right hand $d = x(f_{\text{hi}}) - x(f_{\text{lo}})$, for the left hand $d = x(f_{\text{lo}}) - x(f_{\text{hi}})$. A positive $d$ means the higher-numbered finger is on the side where it belongs; a negative $d$ is a crossing. This one definition handles both hands, ascending and descending motion, and thumb-under versus finger-over.

When the thumb is on a white key and a longer finger on a black key, the finger reaches into the keyboard rather than sideways, so a credit of 2 semitones (finger 3), 1 (fingers 2 and 4) or 0.5 (finger 5) is added to $d$ before the bounds are applied. This is what makes 1-3-1-3 chromatic fingering and 1-2-3 over C-D-E-flat comfortable.

The hand's position is estimated as $\bar{x} = \frac{1}{|S|}\sum_{n \in S} \big(x_{\text{rear}}(n) - o(f_n)\big)$ where $o(f)$ is the relaxed offset of finger $f$ from finger 3 ($-5, -1.5, 0, 1.5, 3.5$ semitones for the right hand, mirrored for the left, scaled by $\sigma$).

## 4. Features

Every feature is dimensionless or in a stated unit so that the default weights are interpretable. Distances inside features are expressed in average semitones, $d / 13.7$ mm.

### 4.1 Span tiers

For a signed span $d$ with bounds $B$, six hinge amounts measure how far $d$ intrudes into each zone, each capped at the width of its zone:

$$
\begin{aligned}
\text{rel\_out} &= \max(0, \min(d, \text{MaxComf}) - \text{MaxRel}) & \text{rel\_in} &= \max(0, \text{MinRel} - \max(d, \text{MinComf}))\\
\text{comf\_out} &= \max(0, \min(d, \text{MaxPrac}) - \text{MaxComf}) & \text{comf\_in} &= \max(0, \text{MinComf} - \max(d, \text{MinPrac}))\\
\text{prac\_out} &= \max(0, d - \text{MaxPrac}) & \text{prac\_in} &= \max(0, \text{MinPrac} - d)
\end{aligned}
$$

With increasing weights per tier the total is a convex piecewise-linear penalty that is zero in the relaxed range, grows gently in the comfortable range, steeply in the practical range and almost prohibitively beyond it. The relaxed-in tier is split into thumb and non-thumb pairs (Parncutt's rule 2 doubles the small-span penalty for non-thumb pairs), and it is dropped for a crossed thumb pair, where a thumb pass is a rotation and is costed by the crossing features instead.

### 4.2 State features $\phi_s(k, s_k)$

Per struck note: use of each finger (five indicator counts, the defaults charge only 4 and 5), weak finger at speed ($[f \in \{4,5\}] \cdot v_k$ with the speed factor of Section 5), thumb, second, fourth and fifth finger on a black key. Per event: the span tiers of Section 4.1 summed over every pair of simultaneously sounding notes, which is the chord playability model.

### 4.3 Transition features $\phi_t(k, s_{k-1}, s_k)$

Each struck note of $E_k$ is paired with the melodically nearest struck note of $E_{k-1}$. For each pair with fingers $f'$ then $f$:

* Repeated key. Same finger: `repeat_same_finger` and `repeat_fast` $= \max(0, \log_2(0.2/\text{IOI}))$. Different finger: `repeat_change`.
* Same finger on a different key: `same_finger_diff_pitch` and its legato-coupled copy weighted by $c_k$ (Section 5).
* Different fingers: the six span tiers, each multiplied by $c_k$. If $d < 0$: a crossing indicator for the thumb pair involved (`cross_1_2` to `cross_1_5`), the crossing amount $|d|$, `cross_thumb_on_black`, `cross_other_on_black` (the passed finger is on a raised black key, which makes the pass easier; its default weight is a discount), or for non-thumb pairs `cross_other` and its amount. Every crossing also contributes `cross_fast` $= c_k v_k$.
* Parncutt's rule 8 (`three_four`) and rule 9 (`four_black_three_white`); the depth change $|z(n) - z(n')|$ between front and rear key surfaces.
* Categorical pair features `pair_{f'}{f}_{up|down}`, 50 indicators with zero default weight. They let the learner absorb idiomatic preferences that no physical feature expresses.

Per transition: the hand-centre shift $|\bar{x}_k - \bar{x}_{k-1}|$ with a 14 mm dead zone (`shift_mm`) and beyond 35 mm (`shift_big`), and the Fitts time pressure of Section 5.

### 4.4 Second-order features $\phi_2(k, s_{k-2}, s_{k-1}, s_k)$

Following the melodic pairing back two events gives fingers $f'', f', f$ on keys $x'', x', x$. Features: the 3-4-5 rule ($\{f'', f', f\} = \{3,4,5\}$), a thumb zigzag (two consecutive thumb crossings in opposite directions), and Parncutt's position-change count and size computed from the span between $x''$ and $x$ with fingers $f''$ and $f$: a full change if outside the comfortable range, half if outside the relaxed range, and the excess beyond the relaxed range as the size. If $f'' = f$ on different keys the hand has moved by $|x - x''|$, which counts as a full change of that size. These are multiplied by $\min(c_{k-1}, c_k)$.

## 5. Time: tempo and articulation

Three quantities per transition come from the parsed timing.

The inter-onset interval $\text{IOI}_k = t_k - t_{k-1}$ gives the speed factor

$$
v_k = \operatorname{clip}\big(\log_2(0.5 / \text{IOI}_k),\ 0,\ 4\big),
$$

zero at quarter notes of 120 BPM and rising by one for every doubling of speed. It multiplies the weak-finger and crossing features and drives the repeated-note features, so at speed the model changes fingers on repeated keys, avoids 4 and 5, and avoids thumb passes where a static position is available.

The gap $g_k = t_k - \max(\text{release of } E_{k-1})$ gives the legato coupling

$$
c_k = \begin{cases} 1 & g_k \le 30\ \text{ms} \\ \exp\!\big(-(g_k - 0.03)/0.25\big) & \text{otherwise,} \end{cases}
$$

which multiplies every span and crossing feature. Notes that overlap or connect must be bridged by the fingers; after a rest the hand relocates and the span constraint fades with the length of the rest.

Hand travel is the distance the hand must relocate beyond what the finger pair can cover from one position: for the same finger the whole key-to-key distance, otherwise $\max(0, |x - x'| - \text{reach})$ with reach the pair's comfortable bound (outward or crossing). Steps and thumb passes within reach are finger movements and cost nothing here. For a genuine leap Fitts's law in Shannon form gives the minimum movement time

$$
T_{\text{move}} = a + b \log_2\!\Big(1 + \frac{D}{W}\Big), \qquad a = 0.05\ \text{s},\ b = 0.10\ \text{s/bit},
$$

with $W$ the width of the landing key, and `fitts_pressure` $= \max(0, T_{\text{move}} - \text{IOI}_k)/0.1$. This is what makes an octave leap prefer 1 to 5 over 1 to 1, and more strongly so the faster the passage.

Tempo enters from the parser: metronome marks anywhere in the score are integrated into a piecewise map from quarter-length offsets to seconds, and the user's own tempo (the speed they intend to practise at) scales the whole map.

## 6. Decoding and learning

The second-order Viterbi recursion runs over pairs of consecutive states,

$$
D_k(p, a) = \min_{h}\big[D_{k-1}(h, p) + Q_k(h, p, a)\big] + T_k(p, a) + S_k(a),
$$

with $S, T, Q$ the weighted feature costs and $T_k(p, a) = +\infty$ when a held note would change finger. With at most 10 states per event the cost is $O(K \cdot 10^3)$, about 2 ms per event in numpy. The decoder is tested against brute-force enumeration on random polyphonic passages.

Because $C = w \cdot \Phi$, human fingerings can be used to learn $w$. For a piece $x$ with human fingering $y^*$, `training/train.py` performs a loss-augmented decode and a passive-aggressive update:

$$
\hat{y} = \arg\min_y \big[w \cdot \Phi(x, y) - \rho\, H(y, y^*)\big], \qquad
\ell = w \cdot \Phi(x, y^*) - w \cdot \Phi(x, \hat{y}) + \rho\, H(\hat{y}, y^*),
$$

$$
\ell > 0:\quad w \leftarrow w + \min\!\Big(C, \frac{\ell}{\|\Phi(x,\hat{y}) - \Phi(x,y^*)\|^2}\Big)\big(\Phi(x,\hat{y}) - \Phi(x,y^*)\big),
$$

where $H$ is the number of notes fingered differently. The update is the smallest change that makes the human fingering cheaper than its strongest rival by the margin, weights are averaged over all steps, and an optional $\ell_2$ pull toward the hand-tuned weights keeps the physical prior. Features are computed once per piece and cached, so an epoch is only the numpy decode.

The intended dataset is PIG (Nakamura, Saito and Yoshii, 2020): 150 classical pieces with fingerings by several pianists, in a text format the loader in `training/pig.py` reads directly. `training/evaluate.py` reports the general and highest match rates used in that literature, so the engine can be compared against published systems (their HMMs reach general match rates in the 60 to 70 percent range, and a rule model like Parncutt's is lower).

## 7. What the defaults produce

With the hand-tuned weights and no training the engine reproduces the standard fingerings for major scales in both hands and directions over one and two octaves, chromatic scales, the C major arpeggio over two octaves (1-2-3-1-2-3-5), root-position and inverted triads (1-3-5, 1-2-5), four-note chords (1-2-3-5), the Alberti bass (5-1-3-1), and it changes fingers on fast repeated notes while keeping one finger on slow ones. On random passages about a third of fingerings differ between a slow and a fast tempo.

## References

Parncutt, R., Sloboda, J. A., Clarke, E. F., Raekallio, M. and Desain, P. (1997). An ergonomic model of keyboard fingering for melodic fragments. *Music Perception* 14(4), 341-382.

Jacobs, J. P. (2001). Refinements to the ergonomic model for keyboard fingering of Parncutt, Sloboda, Clarke, Raekallio, and Desain. *Music Perception* 18(4), 505-511.

Balliauw, M., Herremans, D., Palhazi Cuervo, D. and Sörensen, K. (2017). A variable neighbourhood search algorithm to generate piano fingerings for polyphonic sheet music. *International Transactions in Operational Research* 24(3), 509-535.

Nakamura, E., Saito, Y. and Yoshii, K. (2020). Statistical learning and estimation of piano fingering. *Information Sciences* 517, 68-85. Dataset: https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/

Fitts, P. M. (1954). The information capacity of the human motor system in controlling the amplitude of movement. *Journal of Experimental Psychology* 47(6), 381-391. Shannon form after MacKenzie (1992).

Collins, M. (2002). Discriminative training methods for hidden Markov models. *EMNLP*. Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S. and Singer, Y. (2006). Online passive-aggressive algorithms. *JMLR* 7, 551-585. Taskar, B., Chatalbashev, V., Koller, D. and Guestrin, C. (2005). Learning structured prediction models: a large margin approach. *ICML*.

Key dimensions: Wikipedia, "Musical keyboard" (octave span 164 to 165 mm, white keys about 23.5 mm, black keys about 13.7 mm).
