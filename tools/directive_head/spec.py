"""Generation spec for the standing-directive dataset: languages, topics, categories and templates.

A row's label is 1 when the utterance states a *lasting* directive or preference: something the
speaker (the user) wants to shape future behaviour or recommendations beyond the current request.
Everything else is 0, including utterances that use the same surface markers ("always", "never",
"from now on", "make sure", ...) for one task, idioms, questions, reported speech, narration and
assistant-voice advice.

Splits are by *template* and *topic*, never by random row (see build_data.py):
- a template marked `heldout=True` only ever appears in the test split;
- topics in TEST_TOPICS / VAL_TOPICS only ever appear in the test / validation split.
"""
from __future__ import annotations

LANGS = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese (Brazilian or European)",
    "ja": "Japanese", "zh": "Simplified Chinese", "ko": "Korean", "hi": "Hindi (Devanagari script)",
    "ar": "Modern Standard or colloquial Arabic (Arabic script)",
}

TOPICS = [
    "programming and code review", "cooking and recipes", "travel planning", "fitness and workouts",
    "personal budgeting", "writing emails at work", "music recommendations", "gardening",
    "parenting and kids", "diet and nutrition", "video games", "learning a foreign language",
    "home repair", "car maintenance", "photography", "academic research and citations",
    "contracts and legal paperwork", "customer support replies", "project management", "data analysis and spreadsheets",
    "pets", "clothing and fashion", "movies and TV", "reading and books",
    "team sports", "homework help", "online shopping", "calendar and scheduling",
    "news and current events", "stress and wellbeing", "crafts and DIY", "physics and chemistry",
    "history", "renting and buying a home", "job search and CVs", "social media posts",
    "event and party planning", "astronomy", "coffee and tea", "cycling and running",
]
TEST_TOPICS = {"gardening", "car maintenance", "pets", "astronomy", "crafts and DIY", "renting and buying a home",
               "contracts and legal paperwork", "coffee and tea"}
VAL_TOPICS = {"photography", "history", "team sports", "event and party planning"}

# category -> (label, description, [(template_id, template instruction, heldout)])
CATEGORIES: dict[str, tuple[int, str, list[tuple[str, str, bool]]]] = {
    # ------------------------------------------------------------------ positives
    "pos_marker": (1, "A lasting directive to the assistant that opens with that language's natural equivalent of 'always' or 'never'.", [
        ("pos_marker.always_format", "'Always ...' about how answers should be formatted or written from now on.", False),
        ("pos_marker.never_do", "'Never ...' forbidding something the assistant should not do in any future answer.", False),
        ("pos_marker.please_always", "A polite 'please always/never ...' standing rule.", False),
        ("pos_marker.mid_sentence", "The always/never word appears after a short lead-in (e.g. 'In your answers, always ...', 'Whatever we discuss, never ...').", True),
    ]),
    "pos_scope": (1, "A lasting directive whose permanence is signalled by a time-scope phrase.", [
        ("pos_scope.from_now_on", "Uses that language's equivalent of 'from now on' / 'going forward' / 'from here on'.", False),
        ("pos_scope.in_future", "Uses 'in the future' / 'next time and every time after' / 'for all future chats'.", False),
        ("pos_scope.until_told", "Says the rule holds until the user says otherwise ('until I tell you to stop', 'for good').", True),
    ]),
    "pos_conditional": (1, "A standing conditional rule: whenever some situation recurs, the assistant should act a certain way.", [
        ("pos_conditional.whenever", "'Whenever I ask about X, do Y' or 'every time I ..., ...'.", False),
        ("pos_conditional.if_then", "'If I ever ..., (then) ...' standing rule meant for all future cases.", False),
        ("pos_conditional.when_you", "'When you write/suggest/answer X for me, ...' (a standing rule for that kind of task).", True),
    ]),
    "pos_negation": (1, "A strong lasting prohibition phrased without the plain word 'never'.", [
        ("pos_negation.under_no", "'Under no circumstances ...' / 'at no point ...' / 'not once ...' as a standing rule.", False),
        ("pos_negation.dont_ever", "'Don't ever ...' / 'Stop ... for good' / 'no more ... in your replies'.", False),
        ("pos_negation.avoid", "'Avoid ... in all your answers' / 'Keep ... out of everything you write for me'.", True),
    ]),
    "pos_preference": (1, "The user states a lasting personal preference or taste the assistant should remember for future help.", [
        ("pos_preference.i_prefer", "'I prefer X (over Y)' / 'my preference is ...' about a durable taste.", False),
        ("pos_preference.like_dislike", "'I really like / love / can't stand / hate X' stated as an enduring taste relevant to future suggestions.", False),
        ("pos_preference.constraint", "A durable personal constraint to respect in future suggestions ('I'm vegetarian, so keep recipes meat-free', 'I'm allergic to ...', 'I only use Linux').", False),
        ("pos_preference.rather", "'I'd much rather ... than ...' / 'X works best for me' as a general, lasting preference.", True),
    ]),
    "pos_style": (1, "A lasting instruction about the assistant's style, persona, language or form of address, with no marker words at all.", [
        ("pos_style.address", "How to address the user permanently ('Call me Sam', 'Use the informal you with me').", False),
        ("pos_style.language_units", "Which language, units, currency, date format or spelling to use in every answer.", False),
        ("pos_style.length_tone", "Permanent length/tone/structure of replies ('Keep your replies under three sentences', 'Be blunt with me').", False),
        ("pos_style.remember", "'Remember that ... (for all future conversations)' / 'Keep in mind that I ...' / 'Note for the future: ...'.", True),
    ]),
    "pos_embedded": (1, "A lasting directive or preference embedded in a longer message that also does something else (thanks, a question, a task).", [
        ("pos_embedded.after_thanks", "Thanks or feedback first, then the standing rule ('Great, thanks. By the way, from now on ...').", False),
        ("pos_embedded.with_task", "A one-off task plus a standing rule for the future in the same message.", False),
        ("pos_embedded.rule_list", "A short list (2-4 items) of standing rules for how the assistant should work with the user.", True),
    ]),
    # ------------------------------------------------------------------ negatives
    "neg_one_off": (0, "An instruction that applies only to the current task or moment, even though it may use words like always/never/make sure/don't/from now on.", [
        ("neg_one_off.marker_once", "A one-time instruction containing that language's 'always'/'never'/'make sure'/'don't' that clearly only concerns this single task (e.g. 'Make sure this email sounds polite', 'Wait here until I come back').", False),
        ("neg_one_off.this_reply", "Explicitly scoped to this answer or document only ('For this draft only, ...', 'Just this once, ...').", False),
        ("neg_one_off.plain_task", "An ordinary one-off request with no marker words (translate this, summarize that, fix this bug).", False),
        ("neg_one_off.do_not_now", "'Don't ... yet' / 'Do not send it now' / 'Never mind the second part' about the current task.", True),
    ]),
    "neg_idiom": (0, "Idioms, fixed expressions or exclamations that contain words like always/never/every time/from now on but state no directive.", [
        ("neg_idiom.never_mind", "That language's equivalents of 'never mind', 'never say never', 'better late than never', 'as always'.", False),
        ("neg_idiom.always_phrase", "Casual phrases like 'it's always sunny here', 'always the same story', 'you never know', 'every time I turn around ...'.", False),
        ("neg_idiom.emotional", "Emotional exclamations ('I'll never forget that trip!', 'This always makes me laugh').", True),
    ]),
    "neg_question": (0, "Questions about preferences, rules or habits, not directives.", [
        ("neg_question.ask_pref", "Asking the assistant or someone else what they prefer or whether they always do something.", False),
        ("neg_question.should_i", "'Should I always ...?' / 'Is it OK to never ...?' advice-seeking questions.", False),
        ("neg_question.recall", "Asking whether the assistant remembers a preference or what rule applies ('Did I tell you I prefer ...?').", True),
    ]),
    "neg_reported": (0, "Reported speech: someone else's rule, preference or habit, not a directive to the assistant.", [
        ("neg_reported.someone_says", "'My mother always says ...' / 'My boss told us never to ...'.", False),
        ("neg_reported.other_prefers", "Another person's preference ('My wife prefers window seats', 'The kids hate broccoli').", False),
        ("neg_reported.quoted_rule", "Quoting a sign, policy or rule ('The sign said never feed the animals', 'The policy says always ...').", True),
    ]),
    "neg_narration": (0, "Facts, narration or habits described without asking the assistant to do anything.", [
        ("neg_narration.habit", "The user describing their own past or present habits ('I always walked to school', 'I usually cook on Sundays').", False),
        ("neg_narration.world_fact", "General facts using always/never/every ('The shop is never open on Mondays', 'Water always ...').", False),
        ("neg_narration.story", "A short story or event report from the user's day.", False),
        ("neg_narration.past_pref", "A preference that no longer holds or is merely historical ('I used to love sushi', 'As a kid I hated reading').", True),
    ]),
    "neg_advice": (0, "Assistant-voice or generic advice to the user (e.g. what an assistant reply or article would say), not a user directive.", [
        ("neg_advice.assistant_tip", "An assistant reply giving tips ('Always back up your files before updating', 'Make sure to stretch first').", False),
        ("neg_advice.assistant_ack", "An assistant acknowledging a request or summarizing ('Sure! Here is the revised version ...').", False),
        ("neg_advice.generic_rule", "A general best-practice statement as in a how-to article ('You should never store passwords in plain text').", True),
    ]),
    "neg_temporary": (0, "A preference or request explicitly limited to now, today, or a single occasion.", [
        ("neg_temporary.today", "'Today I'd prefer ...' / 'Tonight I feel like ...' / 'This time let's ...'.", False),
        ("neg_temporary.hypothetical", "Hypotheticals, negated or retracted directives ('If I said always use French, would you remember?', 'I'm not saying you should always ...').", True),
    ]),
    "neg_chat": (0, "Ordinary small talk, greetings or reactions with no directive or preference.", [
        ("neg_chat.smalltalk", "Greetings, small talk, reactions ('Haha nice', 'Good morning! How are you?').", False),
        ("neg_chat.info", "Neutral information sharing ('My flight lands at 6pm', 'I finished the report').", False),
    ]),
}

# Round 2 (added after the first stream evaluation showed the head firing on ordinary task turns): realistic
# negatives at stream-like length and register. They use a per-cell RNG in gen.plan so round-1 cells keep
# their topics, and get more training cells per template (V2_TRAIN_CELLS).
V2_CATS = {"neg_request", "neg_assistant", "neg_share", "pos_context", "neg_opinion", "neg_prompt", "pos_inline_pref"}
V2_TRAIN_CELLS = 8
CATEGORIES.update({
    "neg_request": (0, "A realistic message a user sends an AI assistant asking for help, advice, recommendations, an explanation or a task; often 1-3 sentences of personal context first. 15 to 80 words.", [
        ("neg_request.advice_context", "Personal context, then a request for tips, ideas or recommendations.", False),
        ("neg_request.followup", "A follow-up after the assistant's previous answer: thanks or a reaction, then a new question.", False),
        ("neg_request.task_specs", "A task with constraints that apply to this task only (length limits, number of items, format of this output).", False),
        ("neg_request.own_plans", "The user mentions their own future plans or resolutions ('I'll make sure to visit...', 'I'm going to start running every morning', 'in the future I want to...') and asks something; nothing is asked of the assistant beyond this answer.", False),
        ("neg_request.session_game", "Sets up a role-play, game or persona for this conversation only ('Let's play a game: pretend you are...', 'Act as my interviewer for this practice round').", True),
        ("neg_request.debug", "Pastes or describes a concrete problem (code error, broken appliance, confusing letter) and asks how to fix it.", False),
    ]),
    "neg_assistant": (0, "An AI assistant's reply in a chat: explanations, answers, lists or step-by-step help. 20 to 80 words.", [
        ("neg_assistant.answer", "A direct answer or explanation.", False),
        ("neg_assistant.list", "A short list of options, steps or tips (may use dashes or numbers inline).", False),
        ("neg_assistant.confirm", "Confirms what it will do or summarizes the user's request ('Got it, I'll keep it brief this time.', 'Sure, here is...').", True),
    ]),
    "pos_context": (1, "A realistic 20 to 80 word message with personal or project context that also states a lasting directive or preference the assistant should follow in all future help.", [
        ("pos_context.project_rule", "Project or work context, then a standing rule for every future answer on that project.", False),
        ("pos_context.life_pref", "Personal context (family, health, budget, schedule), then a durable preference to respect in all future suggestions.", False),
        ("pos_context.question_plus_rule", "A question for now plus a sentence that sets a rule for all later answers.", True),
    ]),
    # round 3: the two failure modes the round-2 stream audit showed (opinions/hopes and long single-task
    # prompts flagged; preferences stated inside long requests missed)
    "neg_opinion": (0, "The user reflects, hopes or gives an opinion on society, a topic or the conversation, in 20 to 80 words, often with 'I hope', 'it's important', 'I think we should', 'I wish', 'we must always'. Nothing is asked of the assistant for the future.", [
        ("neg_opinion.hope", "Hopes and wishes about the world or a field ('I hope more companies will...').", False),
        ("neg_opinion.should_society", "'It's important that people always...' / 'We should never...' about society, not about the assistant.", False),
        ("neg_opinion.reaction", "A reaction to the assistant's previous answer with an opinion and maybe a question.", False),
        ("neg_opinion.personal_values", "The user's values or principles about life stated as opinion, not as a rule for the assistant.", True),
    ]),
    "neg_prompt": (0, "A long single-task prompt (30 to 90 words) of the kind people paste into chatbots: it sets up the task and gives formatting or content rules that apply to this one output only.", [
        ("neg_prompt.write_task", "'Write a blog post / email / essay about X. Use 3 headings, keep it under 300 words, do not mention Y.'", False),
        ("neg_prompt.act_as_task", "'I want you to act as a <role>. I will give you <input> and you will <task> for it.' scoped to this one exercise.", False),
        ("neg_prompt.data_task", "Gives data or a document and asks for a transformation (summarize, extract, translate, table) with rules for that output.", False),
        ("neg_prompt.code_task", "Asks for code for one feature with constraints for that code ('use Python 3, no external libraries, add comments').", True),
    ]),
    "pos_inline_pref": (1, "A realistic 20 to 80 word message whose main purpose is a question or request, but which states a durable personal preference in passing that a good assistant should remember for future recommendations.", [
        ("pos_inline_pref.prefer_in_request", "'Can you recommend ...? I prefer X over Y' / '..., I tend to prefer ...' as a stable taste.", False),
        ("pos_inline_pref.constraint_in_request", "A question that mentions a lasting constraint ('I'm lactose intolerant', 'I only fly economy', 'I never drink alcohol').", False),
        ("pos_inline_pref.style_in_request", "A request that also says how they like answers in general ('I like step-by-step explanations', 'I always want sources').", True),
    ]),
    "neg_share": (0, "The user shares personal news, experiences, opinions or feelings at some length without asking for any lasting behaviour. 15 to 80 words.", [
        ("neg_share.experience", "Describes something they did or experienced recently.", False),
        ("neg_share.opinion", "States an opinion about a topic, product or event ('I think the new update is way better').", False),
        ("neg_share.routine", "Describes their routine or schedule as information ('I usually work out Monday and Thursday mornings').", True),
    ]),
})

SYSTEM = (
    "You write realistic, natively fluent chat messages for training a classifier. Write in the requested "
    "language as a native speaker would type it in a chat app (natural word choice, no translation-ese, "
    "no transliteration, the language's normal script). Vary length (3 to 60 words), register (casual, "
    "polite, terse, rambling), punctuation and sentence structure. Do not number the messages and do not "
    "explain them. Never reuse the example phrases from the instructions verbatim."
)

USER_TMPL = """Language: {lang_name}
Topic area: {topic}
Category: {cat_desc}
Template: {tmpl}
Speaker: {speaker}

Write {n} distinct messages that fit the category and template exactly, set in the topic area.
{label_rule}
Return JSON: {{"items": [{{"text": "..."}}]}}."""

LABEL_RULE = {
    1: ("Each message MUST state a lasting directive or preference that the speaker clearly wants applied to "
        "future conversations or answers, not just the current request. Only use the marker or time-scope "
        "phrases the template asks for; do not tack extra persistence phrases onto every message. About a third "
        "of the messages should be short (under 10 words)."),
    0: ("No message may state a lasting directive or preference for the assistant to apply in the future. "
        "Where the template mentions words like always/never/from now on/make sure, use that language's natural "
        "equivalents so the messages are hard negatives for a keyword rule. About a third of the messages "
        "should be short (under 10 words)."),
}


def speaker_for(cat: str, tmpl_id: str) -> str:
    if cat == "neg_advice" and tmpl_id != "neg_advice.generic_rule":
        return "the AI assistant (replying to the user)"
    if cat == "neg_advice":
        return "an article or a knowledgeable person"
    return "the user, talking to their AI assistant"


SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text"],
        "properties": {"text": {"type": "string"}}}}},
}
