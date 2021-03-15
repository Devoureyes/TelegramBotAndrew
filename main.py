from telegram import Update, bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import random
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from bot_config import BOT_CONFIG

HIST_THEME_LEN = 10
hist_theme = []
X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)


def clear_phrase(phrase):
    phrase = phrase.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)
    return result.strip()


# def classify_intent_by_theme(replica, theme=None):
#     # TODO use ML!
#     replica = clear_phrase(replica)
#     for intent, intent_data in BOT_CONFIG['intents'].items():
#         theme_app = None
#         if 'theme_app' in intent_data:
#             theme_app = intent_data['theme_app']
#         if (theme_app is not None and (theme in theme_app or '*' in
#                                    theme_app)) or (
#             theme is None and theme_app is None):
#             for example in intent_data['examples']:
#                 example = clear_phrase(example)
#                 distance = nltk.edit_distance(replica, example)
#                 if distance / len(example) < 0.4:
#                     return intent
# def classify_intents(replica):
#     global hist_theme
#     lev = 0
#     intent = None
#  # Перебор истории тем
#     for theme in hist_theme:
#         intent = classify_intent_by_theme(replica, theme)
#         if intent is not None:
#             break
#         lev += 1
#     if intent is None: # Если по темам не обнаружено намерений, то ищем без темы
#         lev = 0 # Чтобы не очистить историю тем (можно и как вариант очищать, чтобы при непонятках забывать историю)
#         intent = classify_intent_by_theme(replica)
#     else:
#         if lev > 0:
#             hist_theme = hist_theme[lev:] # Перескок на более старую тему, если определеили её
#     if intent is not None:
#         if 'theme_gen' in BOT_CONFIG['intents'][intent]: # Если намерение генерирует новую тему
#             if BOT_CONFIG['intents'][intent]['theme_gen'] not in hist_theme: # И её нет ещё в истории
#                 hist_theme.insert(0,BOT_CONFIG['intents'][intent]['theme_gen']) # То добавляем в историю тему
#                 if (len(hist_theme) > HIST_THEME_LEN):
#                     hist_theme.pop() # Ограничение длины истории тем
#     return intent

def classify_intent(replica):
    replica = clear_phrase(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in BOT_CONFIG['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


with open('Tone.bin') as f:
    tonef = f.read()
tones_str = tonef.split('\n')
allTones = tones_str[0].split(';')[3:]
tones = [tone_str.split(';')[:2] for tone_str in tones_str]
words_with_tones = []
wordsTones = set()

for tone in tones:
    if tone[0] == 'term':
        continue
    wordTone, ton = tone
    wordsTones.add(wordTone)
    words_with_tones.append([wordTone, ton])


def tones(replica):
    emotions = []
    for words in words_with_tones:
        if words[0] in replica:
            emotions.append(words[1])
    if len(emotions) < 1:
        return ["Не определена"]
    else:
        return emotions


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        if responses:
            return random.choice(responses)


with open('dialogues.txt') as f:
    content = f.read()

# dialogues_str = content.split('\n')
# dialogues = [dialogue_str.split('\\')[:2] for dialogue_str in dialogues_str]
dialogues_str = content.split('\n\n')
dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]

dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue

    question, answer = dialogue
    question = clear_phrase(question[:])
    answer = answer[:]

    if question != '' and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  # {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(' '))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

dialogues_structured_cut = {}
for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]


# replica -> word1, word2, word3, ... -> dialogues_structured[word1] + dialogues_structured[word2] + ... -> mini_dataset


def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(' '))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]

    # TODO убрать повторы из mini_dataset
    answers = []  # [[distance_weighted, question, answer]]
    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])
    if answers:
        return min(answers, key=lambda three: three[0])[2]


def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intent': 0, 'generate': 0, 'failure': 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)
    # Answer generation

    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            tones1 = tones(replica)
            tone = "\n(Твоя эмоция " + (', '.join(map(str, tones1))) + ")"
            answer += tone
            stats['intent'] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)

    if answer:
        stats['generate'] += 1
        tones1 = tones(replica)
        tone = "\n(Твоя эмоция " + (', '.join(map(str, tones1))) + ")"
        answer += tone
        return answer

    # берем заглушку
    stats['failure'] += 1
    return get_failure_phrase()


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Привет!')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def run_bot(update: Update, context: CallbackContext) -> None:
    replica = update.message.text
    answer = bot(replica)
    update.message.reply_text(answer)

    file = open("test.txt", "a")
    file.writelines(replica + "\n")
    file.writelines(answer + "\n\n")
    file.close()
    print(stats)


def main():
    """Start the bot."""
    updater = Updater("1680394877:AAEpRMp-VgGgBENq-tElZoRjTV8LEIEq4Hg")

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, run_bot))
    # Start the Bot
    updater.start_polling()
    updater.idle()


main()
