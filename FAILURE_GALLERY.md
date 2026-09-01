# Failure Gallery

Concrete sentences illustrating where the hybrid scorer earns its keep, and
where it (and the rule detector underneath it) still falls short. Pulled
from two sources: the independently hand-verified PROMISE NFR test set
(`data/processed/test_human_verified.csv`, 64 items) and a 10-sentence
out-of-domain stress test written fresh for a fitness-wearable/IoT product —
a domain PROMISE NFR never covers — to check generalization rather than
memorization (`src/out_of_domain_test.py`).

## Where hybrid succeeds and individual methods fail

**1. Rules alone get fooled by sentence structure; hybrid learns the exception.**
> *"The Disputes application shall interface with the Letters application.
> **This** will allow the Disputes application to request letters..."* (REQ-0159, PROMISE)

`This` refers to the whole preceding action, not a specific noun — obvious to
a human, but the rule's antecedent heuristic just counts candidate nouns and
sees several ("application", "Letters application"), so it wrongly flags an
unresolved pronoun (precision 0.25 on this category). The hybrid, trained on
just 2 similar corrected examples elsewhere in the training set, learned to
recognize this "This will allow the X application to..." construction and
correctly suppressed the flag on this held-out sentence — a genuine learned
correction, not a coincidence (see README's Methodology note for how this was
verified).

**2. ML misses vocabulary it didn't see enough of in training; the rule's
lexicon catches it directly — out-of-domain, with no retraining.**
> *"The mobile app shall load **quickly** when launched."* (out-of-domain)

Rules and hybrid both correctly flag `quickly` (vague_quantifier) and its
absence of a number (missing_measurable). The ML-only classifier misses
both — TF-IDF weights learned from ~135 PROMISE sentences don't transfer
cleanly to novel phrasing. Since the rule's lexicon is a plain word list,
not a model fit to one corpus, it fires correctly regardless of domain.

**3. Neither ML nor LLM understands "agent explicitly named" — a real
syntactic case, not just a pattern.**
> *"Battery level readings shall be validated **by the device firmware**
> before being displayed."* (out-of-domain)

This is passive voice, but the actor **is** named. Rules/hybrid correctly
don't flag it (the dependency parse finds the `by X` agent attached to
"validated"). Both ML and LLM incorrectly flag `passive_no_actor` anyway —
they're pattern-matching on "shall be X-ed" surface form without the
syntactic check for whether an agent phrase is actually present.

**4. LLM over-triggers on a case rules/hybrid resolve correctly.**
> *"Only the registered user shall be able to unlock the device using
> **their** fingerprint."* (out-of-domain)

`their` clearly resolves to "the registered user" — rules/hybrid correctly
flag nothing. The LLM baseline (llama3.1:8b, zero-shot) flags both
`passive_no_actor` and `unresolved_pronoun` here, illustrating why a tuned,
task-specific detector beats an off-the-shelf zero-shot prompt on precision,
even against a domain it's never seen.

**5. A genuinely ambiguous pronoun with three candidate referents — the
one case designed to test it — rules/hybrid catch it, ML and LLM don't.**
> *"The app synchronizes with the wearable and the smart scale. **It**
> should update the dashboard within 2 seconds."* (out-of-domain)

"It" could plausibly be the app, the wearable, or the scale. Rules and
hybrid flag `unresolved_pronoun`; ML and LLM both miss it.

## Where hybrid still fails

**1. Over-generalizing the "This"-fix from example 1 above.**
> *"The product shall interface with the Choice Parts System. **This**
> provides the feed of recycled parts data."* (REQ-0204, PROMISE)

Here "This" clearly and unambiguously refers to "the Choice Parts System" —
not ambiguous at all. But the hybrid's correction for clause-referring
"This" (learned from only 2 training examples) generalizes too broadly and
flags this one too. A real trade-off of small-N training: the fix that
correctly generalizes to REQ-0159 above also over-applies here. More
corrected training examples would likely sharpen the boundary.

**2. "by X" specifies a filter criterion, not a responsible actor — a
nuance both the rule and the hybrid miss.**
> *"The list of available follow up actions... must be **filtered by the
> status of the case and the access level of the user**."* (REQ-0197, PROMISE)

Grammatically this reads as passive-with-agent ("filtered by X"), so the
rule doesn't flag it. But "the status of the case" is a filter *criterion*,
not the *actor* doing the filtering (who/what applies the filter is still
unstated). This requires distinguishing instrumental "by" from agentive
"by" — a genuine semantic distinction the dependency parse alone can't make,
and the hybrid inherits the same blind spot since it never saw a training
example correcting this exact pattern.

## Takeaway

The hybrid's wins are real, learned corrections (verified via held-out
generalization, both within-domain and out-of-domain), not just it agreeing
with the rule detector by default. Its failures are honest, explainable, and
traceable to specific causes (small-N training for the pronoun correction;
an instrumental-vs-agentive "by" distinction neither component models) —
exactly the kind of named limitation a rigorous evaluation should surface
rather than hide.
