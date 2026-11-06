# Sentences contradicted by tests 11-13

A running list of sentences in the paper (`paper/main.tex`) and the older READMEs that newer
results contradict or make outdated, each with the new numbers. The paper (`paper/main.tex`) was
updated for every item on 2026-11-06 (line numbers below refer to the version before that edit); the
older READMEs (tests 2, 4 and 6, and the status section of the top-level README) have not been edited. ± is one SE over books unless said otherwise.

## From test 11 (eviction with a probe)

Test 11 scores A2 and a never-read control book C2 frozen with snapshots of the online learner
(Pythia-14M, `full_1e-4`, 20 books).

1. **Eviction is fast, not "slow over 100k tokens".**
   - `paper/main.tex:174` (section title): "Eviction: a book's memory decays slowly"
   - `paper/main.tex:178`: "The comparison is matched, so the memory of A halves over 100k tokens."
   - `test2_eviction/README.md:7-9`: "Memory of A falls slowly ... so memory of A halves over 100k
     tokens, and the "cache size" is over 100k tokens at this learning rate."
   - `README.md` and `experiments_LLM2_cache/README.md`, test 2 row: "slowly: memory of A is +0.038,
     +0.027 and +0.019 ..."

   New: the gain on A2 against the pretrained model is +0.067 ± 0.007 after A1 and halves in about
   5k tokens of other books (+0.023 ± 0.005 at 10k, within noise from 30k). Memory of A (gain on A2
   minus gain on C2) is +0.097 ± 0.013, halves in about 6k tokens, then has a long tail (+0.031 ±
   0.006 at 100k, +0.016 ± 0.008 at 300k). Test 2's slow decline came from scoring A2 online: both
   runs keep learning while reading A2. With frozen probes, test 2's own metric is +0.123, +0.040,
   +0.021 at I = 0, 10k, 100k (3 books), against the published +0.038, +0.027, +0.019.

2. **The eviction test can now separate forgetting from general adaptation.**
   - `paper/main.tex:179`: "How the two runs get there is open: ... consistent both with the A1 run
     keeping A and with it losing A-specific memory while gaining general adaptation at the same
     rate."
   - `paper/main.tex:308`: "Capacity not tested: the eviction test cannot separate forgetting a book
     from adapting to books in general."
   - `test2_eviction/README.md:12` and `:66-68`: "This test cannot tell the two apart." / "This test
     cannot separate them."
   - `README.md:90` (open questions): "the eviction test cannot separate forgetting a book from
     adapting to books in general."
   - README test 2 rows (both READMEs): "... this test cannot tell".

   New: the A1 run loses A. Its frozen gain on A2 falls from +0.067 to +0.003 ± 0.005 at 100k, while
   the gain on a never-read book stays negative (-0.011 to -0.032), so there is no general adaptation
   for a comparison run to catch up with.

3. ~~"Adaptation to books in general" is not supported on unseen books.~~ **Withdrawn after test
   12.** Test 11's control book was negative because its interlude was whole books in order. Test 12
   shows that a learner reading *shuffled* chunks does gain on unseen PG-19 books (see test 12,
   item 6). The test 4 sentences quoted in item 6 stand for the PG-19 domain; only "in general" needs
   qualifying (item 7).

4. **The aggressive single-MLP cache: more early memory is mostly interference.** Not a direct
   contradiction, but test 6/7 text that reads the layer-4 MLP at 3e-3 as a stronger short-range
   cache should add: it has more memory_A than `full_1e-4` (+0.021 ± 0.003 at I = 0) but a lower gain
   on A2 itself (-0.010 ± 0.002), because it loses more on other books (-0.031 ± 0.002 on C2).

## From test 12 (forgetting of general knowledge)

Test 12 scores snapshots of the online learner frozen on WikiText-103 (the same articles raw, 20,480
tokens, and detokenized, 19,456 tokens) and on 6 held-out PG-19 validation books. WikiText numbers
below are detokenized; raw ones are about a third larger, half of the raw forgetting being on
WikiText's formatting markers. Forgetting = snapshot loss - pretrained loss; SE paired over held-out chunks
or books; "last half" = mean over the snapshots in the second half of the stream.

5. **General knowledge outside the domain is not kept.**
   - `README.md:8`: "... new material would gradually evict old material, and general knowledge would
     stay in the weights." (stated as the hypothesis; the paper's `main.tex:48` says the same, "the
     general knowledge in the weights should survive")
   - `paper/main.tex:295`: "General knowledge is only partly kept: tokens not seen recently are
     predicted slightly worse."

   New: on detokenized WikiText, Pythia-14M `full_1e-4` (books in order) loses +0.196 ± 0.006
   nats/token over the last half of the stream (+0.252 ± 0.007 at the end; 4% of the frozen loss of
   4.63). That is about twice its stream gain (+0.100). On the 800k-token stream it levels off at
   +0.23-0.27. "Slightly worse" understates this outside the domain. On held-out PG-19 books there is
   no forgetting (-0.006 in order). (Raw WikiText: +0.312; the extra is formatting.)

6. **Shuffled snapshots gain on unseen books** (answers item 3).
   - `paper/main.tex:130`: "The shuffled learner keeps 61% of the overall gain, which is therefore
     mostly adaptation to old books in general"
   - `test4_checks/README.md:35`: "The shuffled run keeps what adapting to these books in general
     gives."
   - `test4_checks/README.md:67-69`: "the learner still gains +0.061 overall: adapting to PG-19 and
     old books in general does not need the order."

   New: supported for PG-19. Shuffled `full_1e-4` snapshots are better than the pretrained model on
   the 6 held-out books (-0.045 over the last half, -0.052 ± 0.008 at the end), while in-order
   snapshots are not (-0.006). They still lose +0.112 ± 0.004 on detokenized WikiText.

7. **"In general" should be restricted to the domain.** In the sentences of item 6, "books in
   general" is fine, but the gain is not general-purpose: the same shuffled learner loses +0.112
   nats/token on detokenized WikiText (exchange rate 0.55 nats of stream gain per nat lost there).

8. **The dual rule's "good exchange rate" is about the stream, not about forgetting.**
   - `paper/main.tex:297`: "A small fast part on top of the slow learner has a better exchange rate,
     but its gain is small."
   - `README.md:72-73` (status 3): "... adds a short-term cache with a good exchange rate"

   New: measured against held-out forgetting, the dual rule forgets slightly *more* than
   `full_1e-4` (+0.017 ± 0.001 on detokenized WikiText, +0.014 ± 0.001 on books, K = 1). Its
   exchange rate is about the same (0.53 against 0.51). The old sentence refers to gain on recent tokens against
   loss on other stream tokens, which still holds; it should say so.

9. **Capacity: the small model does not forget more per nat of gain** (not contradicted by the
   current text, which already says online learning does not favor the smaller model;
   `paper/main.tex:50` states the hypothesis, "a small model could rely on it more than a larger
   one"). New: 31M at lr 3e-4 forgets more than 14M at 1e-4 (+0.033 ± 0.005 on detokenized
   WikiText, +0.037 ± 0.004 on books, K = 1) for less gain (+0.085 against +0.100). Exchange rate
   0.37 against 0.51 (K1), 0.31 against 0.55 (shuffled). Confounded by the learning rate; test 13
   retunes it.

## From test 13 (update rules across sizes)

Test 13 re-picks plain SGD, Adam and normalized SGD (each tensor moves by the same relative amount
per step) on validation books for 14M, 31M and 70M, and reruns the main stream in order and
shuffled.

10. **70M's weak cache is partly the optimizer, not only size.**
    - `test6/README.md:108-111`: "70M barely gains from online learning at any rate on the check
      books, so this is not a tuning artefact."
    - `paper/main.tex:266`: "at 70M it almost vanishes"
    - `paper/main.tex:268`: "whether its weak locality is about size or about the optimizer remains
      open."
    - `paper/main.tex:307`: "with a non-monotonic locality (flat from 14M to 31M, near zero at 70M)."
    - `README.md:88` (open questions): "plain SGD barely helps Pythia-70M ... Whether that is about
      size or about the optimizer is open"

    New: with plain SGD at 70M the MLPs move about 100x less per step than at 14M (relative update
    1.1e-6 against 1.2e-4), and attention takes 78% of the update. Normalized SGD more than doubles
    70M's gain (+0.057 against +0.025; +0.033 ± 0.002 paired) and raises its locality within 1k from
    +0.016 to +0.084 ± 0.006. That is still far below 14M (+0.373) and 31M (+0.418) under the same
    rule. So "near zero at 70M" is a plain-SGD result; with a normalized update locality falls at
    70M but does not vanish (70M pick at the edge of its grid).

11. **Capacity, with all rules tuned the same way.** Not contradicted (the text already says online
    learning does not favour the smaller model), but now measured directly. Gain ÷ detokenized
    WikiText forgetting, in order: 14M 0.51 / 0.62 / 0.35 (plain SGD / Adam / normalized SGD), 31M
    0.37 / 0.52 / 0.27, 70M 2.68 / 0.46 / 1.48. 14M never pays more than 31M per nat of gain. 70M
    pays much less, but at much smaller steps and gains.

12. **Plain SGD is not a neutral choice of update rule.** Tests 1-12 use plain SGD throughout. Under
    Adam the cache disappears (locality -0.052 to -0.140 within 1k at every size). Under normalized
    SGD it is larger (+0.127 ± 0.005 more locality within 1k than plain SGD at 14M). Sentences that
    present the cache as a property of "online learning" in general (e.g. the README introduction
    and `paper/main.tex:31`) should name the update rule.
