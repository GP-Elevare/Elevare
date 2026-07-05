"""
bloom_pipeline.py
-----------------
Runs on the GLOBAL Python env (no sgcqg needed).
Takes a QA pairs JSON file, runs Bloom-level rephrasing via W&B LLM API,
and outputs the final QA_pairs.json in the format:
[
  {
    "question": "...",
    "reference_answer": "...",
    "student_answer": ""
  },
  ...
]

Usage:
  python bloom_pipeline.py <input_json> [output_json]

Input JSON can be either:
  B) output_100_story_with_context   ->  dict of {story_id: {qa_pairs, context, ...}}
"""

import os
import re
import sys
import json
import time
import threading
from langchain_core.output_parsers import JsonOutputParser

# ── CONFIG ────────────────────────────────────────────────────────────────────
WANDB_API_KEY   = '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'
STAGE2_MODEL    = 'meta-llama/Llama-3.3-70B-Instruct'
MAX_GEN_RETRIES = 3
CONTEXT_WINDOW  = 800

# ── PERSONA / BLOOM DEFINITIONS ───────────────────────────────────────────────
BLOOM_LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']

PERSONA_BLOOM = {
    'beginner':     'Remember',
    'student':      'Understand',
    'practitioner': 'Apply',
    'analyst':      'Analyze',
    'expert':       'Evaluate',
    'innovator':    'Create',
}

BLOOM_STYLE = {
    'Remember':   ('Require the learner to RECALL a specific fact directly stated. '
                   'Pure retrieval, no reasoning. Answer is a short span.'),
    'Understand': ('Require the learner to INTERPRET or EXPLAIN meaning in own words. '
                   'Sense-making: paraphrasing, summarising, explaining why/how.'),
    'Apply':      ('Require the learner to USE the answer in a new practical situation. '
                   'Transfer to a different context.'),
    'Analyze':    ('Require the learner to BREAK DOWN the answer — components, '
                   'implications, comparisons. Decomposition and inference.'),
    'Evaluate':   ('Require the learner to JUDGE the answer against criteria. '
                   'Critical judgement: defending or critiquing based on evidence.'),
    'Create':     ('Require the learner to GENERATE something new — alternative, '
                   'hypothesis, design. Synthesis and original production.'),
}

YESNO_COMPATIBLE_BLOOM = {'Remember', 'Understand', 'Evaluate'}
YESNO_BLOOM_REMAP = {'Apply': 'Understand', 'Analyze': 'Evaluate', 'Create': 'Evaluate'}

_YESNO_YES = {'yes', 'yeah', 'yep', 'correct', 'true', 'right', 'sure', 'indeed'}
_YESNO_NO  = {'no', 'nope', 'not', 'never', 'false', 'wrong', 'incorrect'}

# ── LLM CLIENT ────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _wb_client = OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        api_key=WANDB_API_KEY,
    )
    print('W&B Inference client ready.')
except ImportError:
    print('ERROR: openai package not installed. Run: pip install openai')
    sys.exit(1)

_last_call_time = [0.0]
_throttle_lock  = threading.Lock()

def _throttled_sleep(min_gap=1.0):
    with _throttle_lock:
        elapsed = time.time() - _last_call_time[0]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        _last_call_time[0] = time.time()

def _call_llm(messages, temperature=0.4, max_tokens=512):
    _throttled_sleep()
    for attempt in range(5):
        try:
            resp = _wb_client.chat.completions.create(
                model=STAGE2_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err  = str(e)
            wait = 2 ** attempt
            if 'rate' in err.lower() or '429' in err:
                m = re.search(r'retry.after[\":\s]+(\d+)', err, re.I)
                retry_after = int(m.group(1)) + 1 if m else wait
                print(f'  Rate limited — waiting {retry_after}s...')
                time.sleep(retry_after)
            else:
                print(f'  LLM error (attempt {attempt+1}): {err[:100]}')
                time.sleep(wait)
    raise RuntimeError('LLM Inference failed after 5 attempts.')

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _is_yesno(answer):
    a = answer.strip().lower().rstrip('.,!')
    if a in _YESNO_YES: return True
    if a in _YESNO_NO:  return False
    return None

def _normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def _assign_persona(idx, answer):
    personas = list(PERSONA_BLOOM.keys())
    persona  = personas[idx % len(personas)]
    bloom    = PERSONA_BLOOM[persona]
    # Remap yes/no answers to compatible Bloom levels
    if _is_yesno(answer) is not None:
        if bloom not in YESNO_COMPATIBLE_BLOOM:
            bloom   = YESNO_BLOOM_REMAP.get(bloom, 'Understand')
            persona = next((p for p, b in PERSONA_BLOOM.items() if b == bloom), persona)
    return persona, bloom

# ── REPHRASING ────────────────────────────────────────────────────────────────
# def rephrase_question(context, answer, base_q, bloom, persona, temperature=0.4):
#     """Ask the LLM to rephrase base_q at the given Bloom level."""
#     style = BLOOM_STYLE[bloom]
#     ctx_snippet = context[:CONTEXT_WINDOW] if context else ''

#     system = (
#         f"You are an educational question designer. "
#         f"Rephrase the given question to target the '{bloom}' level of Bloom's taxonomy. "
#         f"The rephrased question must still be answerable with the original answer. "
#         f"Return ONLY the rephrased question, nothing else."
#     )
#     user = (
#         f"Context: {ctx_snippet}\n\n"
#         f"Original question: {base_q}\n"
#         f"Original answer: {answer}\n\n"
#         f"Bloom level '{bloom}' instruction: {style}\n\n"
#         f"Rephrased question ({bloom} level):"
#     )

#     try:
#         result = _call_llm(
#             [{'role': 'system', 'content': system},
#              {'role': 'user',   'content': user}],
#             temperature=temperature,
#             max_tokens=150,
#         )
#         rephrased = result.strip().strip('"').strip("'")
#         # If identical to base, return None so caller can retry
#         if _normalize(rephrased) == _normalize(base_q):
#             return None
#         return rephrased
#     except Exception as e:
#         print(f'  Rephrase failed: {e}')
#         return None


# Persona assignment
def assign_personas(qa_pairs):
    personas = list(PERSONA_BLOOM.keys())
    remapped = 0
    for idx, item in enumerate(qa_pairs):
        persona = personas[idx % len(personas)]
        bloom   = PERSONA_BLOOM[persona]
        if _is_yesno(item.get('answer', '')) is not None:
            if bloom not in YESNO_COMPATIBLE_BLOOM:
                item['bloom_remapped_from'] = bloom
                bloom   = YESNO_BLOOM_REMAP.get(bloom, 'Understand')
                persona = next((p for p, b in PERSONA_BLOOM.items() if b == bloom), persona)
                remapped += 1
        item['persona']     = persona
        item['bloom_level'] = bloom
    if remapped:
        print(f'  Remapped {remapped} yes/no pairs to compatible Bloom levels.')
    return qa_pairs

def _detect_answer_type(span):
    s = span.strip().lower()
    if s in {'yes','no','yeah','nope','correct','wrong','true','false'}: return 'yes/no'
    if re.match(r'^\d{1,4}$', s): return 'number'
    if re.match(r'^(january|february|march|april|may|june|july|august|'
                r'september|october|november|december|\d{4})', s): return 'time/date'
    if re.match(r'^(he|she|they|mr|mrs|dr|[A-Z][a-z]+\s[A-Z][a-z]+)', span): return 'person'
    if re.match(r'^(in |at |on |near |inside |outside |the )', s): return 'location'
    if len(s.split()) > 6: return 'reason/description'
    return 'fact/entity'

def _is_same_as_base(rephrased, base_q):
    """
    True only if the rephrase is genuinely identical to the base question.
    Uses exact match + high token overlap with similar length.
    Avoids false positives where the base is a short fragment (e.g. 'County?',
    'Near what?') that naturally appears as a substring of a valid rephrase.
    """
    r = _normalize(rephrased)
    b = _normalize(base_q)
    if r == b: return True   # exact match after normalisation
    # High token overlap AND rephrase is barely longer → trivially the same
    r_tok = set(r.split())
    b_tok = set(b.split())
    if not r_tok or not b_tok: return r == b
    overlap = len(r_tok & b_tok) / min(len(r_tok), len(b_tok))
    if overlap > 0.85 and len(r.split()) <= len(b.split()) + 2:
        return True
    return False

def _extract_subject(base_q, answer_span):
    q = re.sub(r'[?.!,]', '', base_q).strip()
    q = re.sub(r'^(what|who|where|when|which|how|why|did|does|was|is|were|are)\s+',
               '', q, flags=re.I)
    span_t = set(_normalize(answer_span).split())
    for w in base_q.split():
        if w and w[0].isupper() and w.lower() not in {
            'what','who','where','when','which','how','why','did','the','a','an'
        }:
            return w.rstrip('?.,!')
    remain = [w for w in q.split() if _normalize(w) not in span_t and len(w) > 3]
    return ' '.join(remain[:3]) if remain else answer_span[:40]

# Bloom-level starter phrases — forces the model away from the base question
_BLOOM_STARTERS = {
    'Remember': 'Name / List / Identify / What is the exact / Who specifically / When exactly did',
    'Understand': 'Explain why / How would you describe / What does it mean that',
    'Apply':      'How would you use / In what situation would / Demonstrate how',
    'Analyze':    'What does it reveal that / Why might / What are the implications of',
    'Evaluate':   'Was it justified / Should / How would you judge / To what extent was it right',
    'Create':     'What would happen if / How might things change if / Design a way',
}

_PERSONA_DESC = {
    'beginner':     'is just starting and needs to recall basic facts',
    'student':      'is building understanding and needs to explain concepts',
    'practitioner': 'applies knowledge to solve real problems',
    'analyst':      'breaks down systems and examines implications',
    'expert':       'critically judges approaches against evidence',
    'innovator':    'generates new ideas, designs, and hypotheses',
}

def _build_system_prompt(bloom, persona, answer_span, subject,
                          answer_type, base_q, strict=False):
    starter = _BLOOM_STARTERS.get(bloom, '')
    strict_block = (
        f'STRICT RETRY — your previous output was rejected.\n'
        f'Most likely reason: you returned the base question unchanged.\n'
        f'You MUST produce a genuinely different question that starts with\n'
        f'a {bloom}-level cognitive operation.\n'
        f'Suggested opening words for {bloom}: "{starter}..."\n'
        f'Self-check:\n'
        f'  1. Is your question word-for-word the same as "{base_q}"? -> must be NO\n'
        f'  2. Does it ask about "{subject}"? -> must be YES\n'
        f'  3. Is "{answer_span}" still the core answer? -> must be YES\n'
        f'Rewrite completely if any check fails.\n'
    ) if strict else ''

    _BLOOM_BOUNDARY_RULE = {
        'Remember':   'Structure must be pure retrieval — no explanation, no reasoning, no "why".',
        'Understand': 'Structure must demand explanation or interpretation — not bare recall, not a new scenario.',
        'Apply':      'Structure must place the answer in a NEW situation — not ask what it means (Understand) and not break it down (Analyze).',
        'Analyze':    'Structure must ask what the answer reveals, implies, or consists of — not apply it (Apply) and not judge it (Evaluate).',
        'Evaluate':   'Structure must ask for a judgement or assessment — not what it implies (Analyze) and not a new design (Create). Use: justified, valid, effective, should.',
        'Create':     'Structure must ask for an original design, alternative, or hypothesis — not a judgement (Evaluate) and not mere application (Apply).',
    }

    _BLOOM_DEFINITIONS = {
        'Remember':   ('RETRIEVE facts directly from memory or text. '
                       'The learner must recall a specific piece of information '
                       'exactly as stated. No reasoning, inference, or interpretation. '
                       'Keywords: who, what, when, where, list, name, identify, recall.'),
        'Understand': ('INTERPRET and EXPLAIN meaning in their own words. '
                       'The learner constructs meaning from the information — '
                       'paraphrasing, summarising, classifying, or explaining why/how. '
                       'Keywords: explain, describe, summarise, paraphrase, interpret, why, how.'),
        'Apply':      ('USE knowledge or a procedure in a NEW situation not mentioned in the text. '
                       'The learner must transfer the concept to a different, practical context. '
                       'The question should NOT be answerable by simply re-reading the passage. '
                       'Keywords: use, apply, demonstrate, how would you use, in what situation would.'),
        'Analyze':    ('BREAK DOWN information into parts and examine relationships. '
                       'The learner decomposes the answer — identifying components, '
                       'implications, causes, comparisons, or inferences. '
                       'Keywords: what does it reveal, why might, compare, contrast, '
                       'break down, what is the relationship between, what are the implications of.'),
        'Evaluate':   ('JUDGE or ASSESS based on criteria and standards. '
                       'The learner must defend, critique, or make a judgement about '
                       'the answer using evidence or reasoning. '
                       'Keywords: was it justified, should, assess, evaluate, '
                       'defend, critique, how would you judge, is it valid that.'),
        'Create':     ('GENERATE something NEW by combining or reorganising elements. '
                       'The learner must produce an original idea, hypothesis, alternative, '
                       'or design inspired by the answer — going beyond what is in the text. '
                       'Keywords: what would happen if, imagine, design, propose, '
                       'what if, how might things change if, create, hypothesise.'),
    }

    _BLOOM_EXAMPLES = {
        'Remember': (
            'Example 1:\n'
            '  Context  : "The Amazon River is the largest river by discharge in the world, '
            'flowing through Brazil."\n'
            '  Base Q   : "What river flows through Brazil?"\n'
            '  Answer   : "Amazon River"\n'
            '  Rephrased: "What is the name of the largest river by discharge that flows through Brazil?"\n'
            '  Expected : "The Amazon River flows through Brazil."\n\n'
            'Example 2:\n'
            '  Context  : "Marie Curie won the Nobel Prize in Physics in 1903."\n'
            '  Base Q   : "When did she win it?"\n'
            '  Answer   : "1903"\n'
            '  Rephrased: "In what year did Marie Curie win the Nobel Prize in Physics?"\n'
            '  Expected : "Marie Curie won the Nobel Prize in Physics in 1903."\n\n'
            'Example 3:\n'
            '  Context  : "The attention mechanism is a key component of transformer models."\n'
            '  Base Q   : "What is a key component?"\n'
            '  Answer   : "attention mechanism"\n'
            '  Rephrased: "List the key components of transformer models, including the one '
            'responsible for selective focus."\n'
            '  Expected : "The attention mechanism is a key component of transformer models."\n\n'
            'Example 4:\n'
            '  Context  : "Backpropagation involves a forward pass, loss calculation, and weight update."\n'
            '  Base Q   : "What does backpropagation involve?"\n'
            '  Answer   : "forward pass, loss calculation, and weight update"\n'
            '  Rephrased: "What are the key steps involved in the backpropagation algorithm?"\n'
            '  Expected : "Backpropagation involves a forward pass, loss calculation, and weight update."\n\n'
            'Example 5:\n'
            '  Context  : "A decision tree model requires components such as nodes, branches, '
            'and leaf nodes to make predictions."\n'
            '  Base Q   : "What does a decision tree require?"\n'
            '  Answer   : "nodes, branches, and leaf nodes"\n'
            '  Rephrased: "List and describe the key components of a decision tree model."\n'
            '  Expected : "A decision tree model consists of nodes, branches, and leaf nodes '
            'that together guide predictions."\n\n'
            'Example 6:\n'
            '  Context  : "Stochastic gradient descent uses random sampling to update model weights '
            'at each iteration."\n'
            '  Base Q   : "How does SGD update weights?"\n'
            '  Answer   : "random sampling at each iteration"\n'
            '  Rephrased: "Recall the purpose of stochastic gradient descent in machine learning."\n'
            '  Expected : "SGD uses random sampling to update model weights iteratively, '
            'making optimisation faster on large datasets."'
        ),

        'Understand': (
            'Example 1:\n'
            '  Context  : "Photosynthesis converts sunlight into glucose, which plants use for energy."\n'
            '  Base Q   : "What does photosynthesis do?"\n'
            '  Answer   : "converts sunlight into glucose"\n'
            '  Rephrased: "Explain why photosynthesis is essential for a plant\'s survival."\n'
            '  Expected : "Photosynthesis converts sunlight into glucose, providing plants '
            'with the energy they need to grow and function."\n\n'
            'Example 2:\n'
            '  Context  : "The suspect was released because there was insufficient evidence."\n'
            '  Base Q   : "Was he released?"\n'
            '  Answer   : "yes"\n'
            '  Rephrased: "Explain how the lack of sufficient evidence led to the suspect being released."\n'
            '  Expected : "Yes — the suspect was released because the available evidence was not '
            'strong enough to justify holding him."\n\n'
            'Example 3:\n'
            '  Context  : "Stochastic gradient descent updates weights on a single sample rather '
            'than the full dataset."\n'
            '  Base Q   : "How does it update weights?"\n'
            '  Answer   : "on a single sample"\n'
            '  Rephrased: "Explain how stochastic gradient descent differs from standard gradient '
            'descent in the way it updates model weights."\n'
            '  Expected : "On a single sample — unlike standard gradient descent which uses the '
            'full dataset, SGD updates weights one sample at a time, making it faster but noisier."\n\n'
            'Example 4:\n'
            '  Context  : "A decision tree splits data at each node based on the feature that '
            'reduces entropy the most."\n'
            '  Base Q   : "What does it split on?"\n'
            '  Answer   : "the feature that reduces entropy the most"\n'
            '  Rephrased: "Explain how a decision tree uses entropy to decide where to split data at each node."\n'
            '  Expected : "The feature that reduces entropy the most — a decision tree selects this '
            'at each node to create the most ordered, pure subgroups."\n\n'
            'Example 5:\n'
            '  Context  : "Entropy measures the impurity of a node when classifying Indian festivals '
            'by date and location."\n'
            '  Base Q   : "What does entropy measure?"\n'
            '  Answer   : "impurity of a node"\n'
            '  Rephrased: "Explain how the concept of entropy is used in decision tree construction."\n'
            '  Expected : "Impurity of a node — entropy quantifies how mixed the classes are at each '
            'split point, and the tree reduces this impurity to create purer subgroups."\n\n'
            'Example 6:\n'
            '  Context  : "Feature importance scores in gradient boosted trees indicate which '
            'variables most influence predictions."\n'
            '  Base Q   : "What do feature importance scores indicate?"\n'
            '  Answer   : "which variables most influence predictions"\n'
            '  Rephrased: "Explain the concept of feature importance in gradient boosted tree models."\n'
            '  Expected : "Which variables most influence predictions — feature importance scores '
            'quantify each variable\'s contribution so practitioners can focus on the most '
            'decisive factors and discard irrelevant ones."'
        ),

        'Apply': (
            'Example 1:\n'
            '  Context  : "Cotton lived in a barn with her mother and five sisters."\n'
            '  Base Q   : "Where did she live?"\n'
            '  Answer   : "in a barn"\n'
            '  Rephrased: "In what real-world situation would you apply the same principle of '
            'communal living that Cotton\'s family used in the barn?"\n'
            '  Expected : "In a barn — or its equivalent — communal living like Cotton\'s '
            'arrangement could be applied in co-housing communities where families share space '
            'and resources."\n\n'
            'Example 2:\n'
            '  Context  : "Reginald used a flashlight when the power went out during the storm."\n'
            '  Base Q   : "What device did they use?"\n'
            '  Answer   : "Flashlight"\n'
            '  Rephrased: "How would you use a flashlight in an emergency situation where the '
            'power grid fails across an entire neighbourhood?"\n'
            '  Expected : "A flashlight could be used to navigate safely, signal for help, '
            'and perform basic tasks during a neighbourhood-wide power outage."\n\n'
            'Example 3:\n'
            '  Context  : "Indian Premier League cricket matches depend on team composition, '
            'location, and weather conditions."\n'
            '  Base Q   : "What do matches depend on?"\n'
            '  Answer   : "team composition, location, and weather conditions"\n'
            '  Rephrased: "Using a dataset of IPL cricket matches, demonstrate how you would '
            'apply a decision tree model to predict match outcomes."\n'
            '  Expected : "Team composition, location, and weather conditions — these would '
            'serve as input features in a decision tree to classify likely match outcomes."\n\n'
            'Example 4:\n'
            '  Context  : "Lemmatization reduces each word to its base dictionary form."\n'
            '  Base Q   : "What does lemmatization do?"\n'
            '  Answer   : "reduces each word to its base dictionary form"\n'
            '  Rephrased: "With the sentence \'The quick brown fox jumps\', '
            'demonstrate how you would apply lemmatization to each word."\n'
            '  Expected : "Reduces each word to its base dictionary form — for example, '
            '\'jumps\' becomes \'jump\' and \'running\' becomes \'run\'."\n\n'
            'Example 5:\n'
            '  Context  : "Air quality in major Indian cities is measured using PM2.5 levels, '
            'temperature, and humidity."\n'
            '  Base Q   : "What is air quality measured with?"\n'
            '  Answer   : "PM2.5 levels, temperature, and humidity"\n'
            '  Rephrased: "Demonstrate how you would apply decision tree regression to predict '
            'the air quality index in Indian cities."\n'
            '  Expected : "PM2.5 levels, temperature, and humidity — these features would be '
            'used as inputs in a decision tree regression model trained on historical AQI data."\n\n'
            'Example 6:\n'
            '  Context  : "Customer churn in Indian telecom companies is driven by factors such as '
            'call quality, pricing, and customer service."\n'
            '  Base Q   : "What drives customer churn?"\n'
            '  Answer   : "call quality, pricing, and customer service"\n'
            '  Rephrased: "Given a dataset of customer churn for an Indian telecom company, apply '
            'a decision tree model to identify at-risk customers."\n'
            '  Expected : "Call quality, pricing, and customer service — a decision tree trained '
            'on these features could flag at-risk customers and reveal the primary churn drivers."'
        ),
        'Analyze': (
            'Example 1:\n'
            '  Context  : "Kendra and Quinton travel to school every day. Kendra lives further '
            'from the bus stop than Quinton."\n'
            '  Base Q   : "Does Quinton live further from the bus stop?"\n'
            '  Answer   : "No"\n'
            '  Rephrased: "What does it reveal about their daily routines that Kendra lives '
            'further from the bus stop than Quinton?"\n'
            '  Expected : "No, Quinton lives closer — this reveals that Kendra likely spends '
            'more time commuting each day, which could affect her schedule and energy levels."\n\n'
            'Example 2:\n'
            '  Context  : "Der Spiegel published photos of American soldiers posing over the '
            'bodies of dead Afghans."\n'
            '  Base Q   : "What were the soldiers doing in the photos?"\n'
            '  Answer   : "posing over the bodies of dead Afghans"\n'
            '  Rephrased: "What does the soldiers\' decision to pose over the bodies reveal '
            'about military discipline and ethical conduct?"\n'
            '  Expected : "The act of posing over the bodies of dead Afghans reveals a serious '
            'lapse in military ethics and discipline, suggesting inadequate oversight and a '
            'dehumanising attitude toward casualties."\n\n'
            'Example 3:\n'
            '  Context  : "A gradient boosted tree model trained on Indian election data showed '
            'unstable loss curves at high learning rates."\n'
            '  Base Q   : "What did the loss curves show?"\n'
            '  Answer   : "unstable loss curves at high learning rates"\n'
            '  Rephrased: "What does the unstable convergence behavior at high learning rates '
            'reveal about the model\'s sensitivity to this hyperparameter?"\n'
            '  Expected : "The unstable loss curves at high learning rates reveal that the model '
            'is highly sensitive to this hyperparameter, suggesting a need for careful tuning '
            'or a learning rate scheduler."\n\n'
            'Example 4:\n'
            '  Context  : "A CNN trained on Indian facial datasets uses convolutional layers '
            'to extract features like edges, textures, and shapes."\n'
            '  Base Q   : "What does the CNN extract?"\n'
            '  Answer   : "features like edges, textures, and shapes"\n'
            '  Rephrased: "What does the progression from edges to textures to shapes across '
            'CNN layers reveal about how the model builds its understanding of faces?"\n'
            '  Expected : "Extracting features like edges, textures, and shapes in a hierarchical '
            'progression reveals that early layers detect low-level patterns while deeper layers '
            'combine them into meaningful facial representations."\n\n'
            'Example 5:\n'
            '  Context  : "A decision tree categorises Indian dishes as spicy or non-spicy '
            'based on ingredients such as chilli, pepper, and cumin."\n'
            '  Base Q   : "How does it categorise dishes?"\n'
            '  Answer   : "based on ingredients such as chilli, pepper, and cumin"\n'
            '  Rephrased: "Analyze how the decision tree makes its classifications at each '
            'node and what factors contribute most significantly."\n'
            '  Expected : "The tree classifies dishes based on ingredients such as chilli, '
            'pepper, and cumin — likely splitting on the highest-heat ingredient first, '
            'revealing that capsaicin-heavy components are the most discriminative features."\n\n'
            'Example 6:\n'
            '  Context  : "A gradient boosted tree classifier trained on Indian stock market data '
            'assigned high importance to trading volume and P/E ratio."\n'
            '  Base Q   : "What did the classifier assign importance to?"\n'
            '  Answer   : "trading volume and P/E ratio"\n'
            '  Rephrased: "Analyze the feature importance scores — what do the most critical '
            'factors reveal about the drivers of stock price fluctuations?"\n'
            '  Expected : "High importance assigned to trading volume and P/E ratio implies '
            'that market activity and valuation sentiment are the primary drivers, suggesting '
            'the model captures both momentum and fundamental signals."'
        ),
        'Evaluate': (
            'Example 1:\n'
            '  Context  : "Dennis Farina died at 69. He was known for his roles in Law & Order."\n'
            '  Base Q   : "Is he still alive?"\n'
            '  Answer   : "No"\n'
            '  Rephrased: "To what extent was it justified that Dennis Farina is remembered '
            'primarily for his TV roles, given the full scope of his career?"\n'
            '  Expected : "No — it is not fully justified, because while his TV roles were '
            'iconic, dismissing his film work would undervalue the full breadth of his career."\n\n'
            'Example 2:\n'
            '  Context  : "Green tea may prevent heart disease. Fermentation in black tea '
            'can negate its healthy qualities."\n'
            '  Base Q   : "What good thing does tea do for health?"\n'
            '  Answer   : "may prevent heart disease"\n'
            '  Rephrased: "To what extent is green tea a reliable health intervention, '
            'given that it may prevent heart disease but evidence varies?"\n'
            '  Expected : "Green tea may prevent heart disease, but this should be treated '
            'with caution — the word \'may\' signals limited certainty, making it a supplement '
            'to rather than a replacement for proven medical interventions."\n\n'
            'Example 3:\n'
            '  Context  : "A decision tree classifier was used to predict crop diseases '
            'in India with 78% accuracy."\n'
            '  Base Q   : "How accurate was it?"\n'
            '  Answer   : "78% accuracy"\n'
            '  Rephrased: "How effective is a 78% accuracy rate for a crop disease classifier '
            'in an Indian agricultural context — is this performance acceptable?"\n'
            '  Expected : "78% accuracy is likely insufficient for high-stakes agricultural '
            'decisions where misclassification leads to crop loss — higher precision and '
            'recall would be needed before deployment."\n\n'
            'Example 4:\n'
            '  Context  : "Reinforcement learning with human feedback was used to align a '
            'conversational AI for Indian language support."\n'
            '  Base Q   : "What was used to align it?"\n'
            '  Answer   : "reinforcement learning with human feedback"\n'
            '  Rephrased: "Should reinforcement learning with human feedback be the standard '
            'approach for aligning conversational AI in low-resource Indian language settings?"\n'
            '  Expected : "Reinforcement learning with human feedback is effective, but its '
            'suitability depends on annotation budget and annotator quality — both are often '
            'constrained for low-resource Indian languages."\n\n'
            'Example 5:\n'
            '  Context  : "A decision tree model was evaluated for predicting loan defaults '
            'across various Indian banks using precision, recall, and F1 score."\n'
            '  Base Q   : "How was the model evaluated?"\n'
            '  Answer   : "precision, recall, and F1 score"\n'
            '  Rephrased: "How effective is a decision tree for predicting loan defaults — '
            'what do precision, recall, and F1 reveal about its reliability for this task?"\n'
            '  Expected : "Using precision, recall, and F1 score reveals that in this '
            'imbalanced context recall matters most — missing a default is costlier than '
            'a false alarm, so the model should be assessed primarily on recall."\n\n'
            'Example 6:\n'
            '  Context  : "A machine learning model was developed to forecast energy consumption '
            'in Indian households across training, validation, and testing phases."\n'
            '  Base Q   : "What did the model forecast?"\n'
            '  Answer   : "energy consumption in Indian households"\n'
            '  Rephrased: "How would you assess the reliability of a model forecasting energy '
            'consumption in Indian households if validation accuracy is much higher than test accuracy?"\n'
            '  Expected : "A model forecasting energy consumption in Indian households with '
            'a large validation-to-test accuracy gap is likely overfitting — its real-world '
            'reliability should be considered poor until regularisation is applied."'
        ),

        'Create': (
            'Example 1:\n'
            '  Context  : "Asta and his friends found a note inside a bottle in the ocean."\n'
            '  Base Q   : "What was in it?"\n'
            '  Answer   : "a note"\n'
            '  Rephrased: "What would happen if Asta designed a new underwater communication '
            'system inspired by the bottle-and-note discovery?"\n'
            '  Expected : "A note — but reimagined as a waterproof digital capsule that Asta '
            'could design to float between ocean zones, carrying community messages in a '
            'living undersea postal network."\n\n'
            'Example 2:\n'
            '  Context  : "Tea was created by accident. Great Britain consumes the most tea."\n'
            '  Base Q   : "How was tea created?"\n'
            '  Answer   : "by accident"\n'
            '  Rephrased: "Design a modern beverage discovery process inspired by how tea was '
            'accidentally created — what steps would encourage productive accidents?"\n'
            '  Expected : "By accident — a modern discovery process could formalise this by '
            'including unstructured experimentation sessions and randomisation trials that '
            'treat unexpected combinations as potential innovations."\n\n'
            'Example 3:\n'
            '  Context  : "A gradient boosted tree model was designed to assess credit risk '
            'for loan applicants at Indian banks."\n'
            '  Base Q   : "What was it designed to assess?"\n'
            '  Answer   : "credit risk"\n'
            '  Rephrased: "Design a gradient boosted tree model to assess credit risk for '
            'individuals applying for loans at Indian banks."\n'
            '  Expected : "Credit risk — the model could use features like income, repayment '
            'history, and employment stability, with thresholds tuned to minimise defaults '
            'while avoiding over-rejection of valid applicants."\n\n'
            'Example 4:\n'
            '  Context  : "Transfer learning allows a pre-trained model to be adapted for '
            'new tasks with limited data."\n'
            '  Base Q   : "What does transfer learning allow?"\n'
            '  Answer   : "a pre-trained model to be adapted for new tasks with limited data"\n'
            '  Rephrased: "How might things change if you designed a transfer learning strategy '
            'for classifying Indian wildlife species using limited annotated images?"\n'
            '  Expected : "A pre-trained model adapted for new tasks with limited data — '
            'fine-tuning on a small curated Indian wildlife dataset by replacing only the '
            'final layers would significantly reduce the compute and annotation needed."\n\n'
            'Example 5:\n'
            '  Context  : "A decision tree can predict the success of Indian startups based on '
            'sector, funding stage, and founding team experience."\n'
            '  Base Q   : "What can predict startup success?"\n'
            '  Answer   : "sector, funding stage, and founding team experience"\n'
            '  Rephrased: "Design a decision tree model to predict the success of Indian startups '
            'across technology, healthcare, and education sectors."\n'
            '  Expected : "Sector, funding stage, and founding team experience — the tree could '
            'split first by sector viability, then by funding adequacy, with leaf nodes '
            'predicting high, medium, or low success probability."\n\n'
            'Example 6:\n'
            '  Context  : "Anomalies in electricity consumption patterns in rural Indian villages '
            'can signal theft or equipment failure."\n'
            '  Base Q   : "What can anomalies signal?"\n'
            '  Answer   : "theft or equipment failure"\n'
            '  Rephrased: "Design a machine learning pipeline to detect anomalies in electricity '
            'consumption patterns in rural Indian villages."\n'
            '  Expected : "Theft or equipment failure — the pipeline could use time-series '
            'features with an isolation forest to flag anomalies, with alert thresholds '
            'calibrated to minimise false positives in low-consumption areas."'
        ),
    }

    bloom_def = _BLOOM_DEFINITIONS.get(bloom, '')
    bloom_ex  = _BLOOM_EXAMPLES.get(bloom, '')

    return (
        f"You are an expert question designer who crafts questions about any topic — "
        f"everyday stories, news, science, history, or fiction — at precisely the "
        f"{bloom} level of Bloom's revised taxonomy. No higher, no lower.\n\n"

        f'## YOUR TASK\n'
        f'Rewrite the base question to demand {bloom.upper()}-level cognitive processing '
        f"from a '{persona}' learner.\n\n"

        f'## BLOOM LEVEL: {bloom.upper()}\n'
        f'{bloom_def}\n\n'

        f'## COGNITIVE PROCESS DETAIL\n'
        f'{BLOOM_STYLE[bloom]}\n\n'

        f'## FEW-SHOT EXAMPLES FOR {bloom.upper()}\n'
        f'{bloom_ex}\n\n'

        f'## LOCKED (must not change)\n'
        f'  Answer span    = "{answer_span}"\n'
        f'  Subject entity = "{subject}"\n'
        f'  Answer type    = {answer_type}\n\n'

        f'## MUST CHANGE\n'
        f'  Reframe the question so a {persona} learner — someone who {_PERSONA_DESC[persona]} — '
        f'would naturally ask it at the {bloom} cognitive level.\n'
        f'  The wording, cognitive operation, and register must all change from the base question.\n\n'

        f'## SUGGESTED OPENING for {bloom}\n'
        f'  "{starter}..."\n'
        f'  Use this as inspiration. Your question must begin with this kind of '
        f'cognitive framing, not a simple recall question.\n\n'

        f'{strict_block}'

        f'## CRITICAL RULES\n'
        f'1. Your rephrased_question MUST be different from "{base_q}" — '
        f'returning it unchanged is a failure.\n'
        f'2. {_BLOOM_BOUNDARY_RULE[bloom]}\n'
        f'3. Do NOT introduce new factual claims not in the context. '
        f'For {bloom}: reasoning, inference, and hypotheticals beyond the text are expected — '
        f'but any concrete facts you add must be derivable from the context.\n'
        f'4. One or two concise natural sentences ending with "?". '
        f'Higher Bloom levels (Analyze, Evaluate, Create) may use a brief setup clause followed by the question.\n'
        f'5. expected_answer: must clearly contain or reference "{answer_span}" — '
        f'do not replace it with elaboration. The original answer must remain '
        f'identifiable in your response. You may paraphrase it minimally, '
        f'then extend naturally with {bloom}-level reasoning in whatever '
        f'phrasing sounds most natural for this question and context.\n\n'
        f'6. For Apply level: the new situation must still require "{answer_span}" as its core answer — '
        f'do NOT invent a scenario where a different answer would emerge.\n\n'
        f'7. For Evaluate: use judgment words (justified, right, effective, valid, should) — '
        f'never use "conclude" or "infer" as these belong to Analyze.\n\n'

        f'## THINK BEFORE YOU WRITE\n'
        f'Before generating, silently verify:\n'
        f'  a) What cognitive operation does {bloom} require? (not just recall)\n'
        f'  b) Does my question force that operation, or does it slip into a lower level?\n'
        f'  c) Is "{answer_span}" clearly present or referenced in my expected_answer — '
        f'not replaced by elaboration?\n\n'

        f'## OUTPUT FORMAT\n'
        f'Return ONLY valid JSON (no markdown fences, no extra text):\n'
        f'{{"rephrased_question": "string", "expected_answer": "string"}}'
    )


def rephrase_question(context, answer_span, base_q, bloom, persona,
                      strict=False, temperature=0.4):
    subject     = _extract_subject(base_q, answer_span)
    answer_type = _detect_answer_type(answer_span)
    system      = _build_system_prompt(bloom, persona, answer_span,
                                        subject, answer_type, base_q, strict)
    yesno       = _is_yesno(answer_span)
    yesno_note  = ''
    if yesno:
        patterns = {
            'Remember':   f'Ask directly whether {subject} did/was/is something. Answer: "{answer_span}".',
            'Understand': f'Ask WHY or HOW the answer is "{answer_span}".',
            'Evaluate':   f'Ask whether it was right/justified that the answer is "{answer_span}".',
        }
        pat = patterns.get(bloom, patterns['Understand'])
        yesno_note = f'YES/NO GUIDANCE: answer is "{answer_span.upper()}". Pattern: {pat}\n'

    user = (
        f'Context:\n{context[:CONTEXT_WINDOW]}\n\n'
        f'Base question  : {base_q}\n'
        f'Answer span    : {answer_span}\n'
        f'Subject entity : {subject}\n'
        f'Answer type    : {answer_type}\n'
        f'{yesno_note}\n'
        f'Rewrite as {bloom}-level for a "{persona}" learner.\n'
        f'Your question MUST be different from "{base_q}".\n'
        f'Keep answer span, subject, answer type LOCKED.\n'
        # REMINDER: semantic presence, not forced format
        f'REMINDER: your expected_answer must clearly contain or reference "{answer_span}" — '
        f'it must remain identifiable in your answer. Express it naturally; '
        f'do not replace it entirely with elaboration.'
    )

    raw =_call_llm(
        [{'role': 'system', 'content': system},
         {'role': 'user',   'content': user}],
        temperature=temperature, max_tokens=512,
    )

    parser = JsonOutputParser()
    try:
        parsed = parser.parse(raw)
        rq = parsed.get('rephrased_question', base_q)
        ea = parsed.get('expected_answer', answer_span)

        if _is_same_as_base(rq, base_q):
            return None

        return {
            'rephrased_question':    rq,
            'expected_answer':       ea,          # model's full anchored answer
            'subject_entity':        subject,
            'answer_type':           answer_type,
        }
    except Exception:
        return None




# ── LOAD INPUT ────────────────────────────────────────────────────────────────
def load_input(filepath):
    """
    Accepts two formats:
      A) List: [{question, reference_answer, student_answer}, ...]  <- from QG_pipeline
      B) Dict: {story_id: {qa_pairs: [{question, answer}], context, ...}}  <- from CoQA output
    Returns list of {question, answer, context}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = []

    if isinstance(data, list):
        # Format A — from QG_pipeline output
        print(f'Detected Format A (QG_pipeline list): {len(data)} items')
        for item in data:
            items.append({
                'question': item.get('question', ''),
                'answer':   item.get('answer') or item.get('reference_answer', ''),
                'context':  item.get('context', ''),
            })

    elif isinstance(data, dict):
        # Format B — from CoQA/SG-CQG output
        print(f'Detected Format B (story dict): {len(data)} stories')
        for story_id, story in data.items():
            context  = story.get('context', '')
            qa_pairs = story.get('qa_pairs', [])
            for pair in qa_pairs:
                items.append({
                    'question': item.get('question', ''),
                    'answer':   item.get('answer') or item.get('reference_answer', ''),
                    'context':  item.get('context', ''),
                })
    else:
        raise ValueError(f'Unknown input format: {type(data)}')

    print(f'Total QA pairs loaded: {len(items)}')
    return items

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_bloom_pipeline(input_path, output_path='QA_pairs.json'):
  
    print(f"CHECK -> Q:")

    items = load_input(input_path)

  

    output = []
    for idx, item in enumerate(items):
        base_q  = item['question']
        answer  = item['answer']
        context = item['context']

        if not base_q or not answer:
            continue

        persona, bloom = _assign_persona(idx, answer)
        print(f'\n[{idx+1}/{len(items)}] Persona={persona} | Bloom={bloom}')
        print(f'  base_q : {base_q[:80]}')
        print(f'  answer : {answer[:60].encode("cp1252", errors="ignore").decode("cp1252")}')

        # Try rephrasing up to MAX_GEN_RETRIES times
        rephrased = None
        for attempt in range(1, MAX_GEN_RETRIES + 1):
            temp = [0.4, 0.6, 0.8][min(attempt - 1, 2)]
            rephrased = rephrase_question(context, answer, base_q, bloom, persona, temperature=temp)
            if rephrased:
                print(f'  rephrased ({bloom}): {rephrased["rephrased_question"][:80]}')
                break
            print(f'  attempt {attempt} failed or identical, retrying...')

        # Fallback to base question if all retries failed
        if rephrased:
            final_q = rephrased["rephrased_question"]
        else:
            final_q = base_q
            print('  WARNING: Using base question as fallback.')

        output.append({
            # "orignal question": base_q,
            'question': final_q,
            'reference_answer': answer,
            'student_answer': ''
        })

    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nDone! {len(output)} QA pairs saved to {output_path}')
    return output

# # ── ENTRY POINT Testing ───────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print('Usage: python bloom_pipeline.py <input_json> [output_json]')
#         print('  input_json  : QA_pairs.json from QG_pipeline OR CoQA output dict')
#         print('  output_json : output file (default: QA_pairs_bloom.json)')
#         sys.exit(1)

#     input_path  = sys.argv[1]
#     output_path = sys.argv[2] if len(sys.argv) > 2 else 'QA_pairs_bloom.json'

#     result = run_bloom_pipeline(input_path, output_path)

#     # Print JSON summary for Node.js to parse
#     print(json.dumps({
#         'message':     'Bloom rephrasing complete',
#         'output_path': output_path,
#         'total':       len(result),
#         'result':      result,
#     }))