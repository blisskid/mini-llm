"""
prepare_qa_data.py - Generates a diverse conversational Q&A dataset for MiniGPT.
Formats data as:
User: <Question>
Assistant: <Answer>
<|end|>
"""

import random

QA_PAIRS = [
    # Greetings & Identity
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What can I help you with?"),
    ("Hey", "Hello! Good to see you. How can I assist you?"),
    ("Good morning", "Good morning! I hope you have a great day. How can I assist you?"),
    ("Good afternoon", "Good afternoon! How can I help you today?"),
    ("Good evening", "Good evening! What can I do for you tonight?"),
    ("How are you?", "I am doing well, thank you! How can I help you today?"),
    ("How are you doing?", "I am doing great! Ready to answer your questions."),
    ("Who are you?", "I am MiniGPT, a small language model trained from scratch in PyTorch."),
    ("What is your name?", "My name is MiniGPT."),
    ("What are you?", "I am a lightweight decoder-only Transformer language model."),
    ("What can you do?", "I can answer questions, explain concepts, and chat with you."),
    ("Who created you?", "I was trained locally using PyTorch on Apple Silicon."),
    ("Thank you", "You are very welcome! Let me know if you need anything else."),
    ("Thanks", "Glad to help! Any other questions?"),
    ("Bye", "Goodbye! Have a wonderful day."),
    ("Goodbye", "Farewell! Feel free to ask more questions anytime."),

    # World Facts & Geography
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of Russia?", "The capital of Russia is Moscow."),
    ("What is the largest country by area?", "Russia is the largest country in the world by land area."),
    ("What is the most populous country?", "India and China are the most populous countries in the world."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the highest mountain in the world?", "Mount Everest is the highest mountain above sea level."),
    ("What is the longest river in the world?", "The Nile River is traditionally considered the longest river in the world."),

    # Science & Nature
    ("Why is the sky blue?", "The sky looks blue because molecules in Earth's atmosphere scatter sunlight in all directions, and blue light scatters more than other colors because it travels as shorter, smaller waves."),
    ("What is water made of?", "Water is a chemical compound consisting of two hydrogen atoms and one oxygen atom (H2O)."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What is the closest star to Earth?", "The Sun is the closest star to Earth."),
    ("How many planets are in the solar system?", "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet because of iron oxide on its surface."),
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants and organisms transform light energy into chemical energy."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air during photosynthesis."),
    ("What gas do humans breathe in to live?", "Humans breathe in oxygen to survive."),
    ("What is gravity?", "Gravity is a fundamental force of attraction that pulls objects toward one another."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second, or about 300,000 km/s."),
    ("How many continents are there on Earth?", "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."),
    ("What is the boiling point of water?", "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit."),
    ("What is the freezing point of water?", "The freezing point of water is 0 degrees Celsius or 32 degrees Fahrenheit."),

    # Math & Logic
    ("What is 1 + 1?", "1 + 1 = 2."),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What is 3 + 3?", "3 + 3 = 6."),
    ("What is 4 + 4?", "4 + 4 = 8."),
    ("What is 5 + 5?", "5 + 5 = 10."),
    ("What is 10 + 10?", "10 + 10 = 20."),
    ("What is 5 * 5?", "5 multiplied by 5 is 25."),
    ("What is 6 * 7?", "6 multiplied by 7 is 42."),
    ("What is 10 / 2?", "10 divided by 2 is 5."),
    ("What is 100 - 50?", "100 minus 50 is 50."),
    ("Is 10 greater than 5?", "Yes, 10 is greater than 5."),
    ("Is 3 an odd or even number?", "3 is an odd number."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("What is the square root of 100?", "The square root of 100 is 10."),

    # Technology & AI
    ("What is Python?", "Python is a popular, high-level, general-purpose programming language known for its readability and ease of use."),
    ("What is PyTorch?", "PyTorch is an open-source machine learning framework developed primarily by Meta AI, widely used for deep learning."),
    ("What is a Transformer in AI?", "A Transformer is a deep learning neural network architecture introduced in 2017 that relies on self-attention mechanisms to model sequential data."),
    ("What is an LLM?", "An LLM (Large Language Model) is a deep neural network trained on massive amounts of text to understand and generate human language."),
    ("What is self-attention?", "Self-attention is a mechanism in Transformers that allows each token in a sequence to look at and weigh the importance of all other tokens."),
    ("What is an API?", "An API (Application Programming Interface) allows different software applications to communicate and exchange data with each other."),
    ("What is an operating system?", "An operating system is system software that manages computer hardware, software resources, and provides common services for computer programs."),
    ("What is CPU?", "A CPU (Central Processing Unit) is the primary component of a computer that executes program instructions."),
    ("What is GPU?", "A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly process mathematical calculations, especially for parallel computing and AI."),
    ("What does HTML stand for?", "HTML stands for HyperText Markup Language."),
    ("What is a function in programming?", "A function is a reusable block of code that takes inputs, performs specific actions, and returns an output."),
    ("What is a variable in programming?", "A variable is a named storage location in computer memory that holds a value."),

    # General Knowledge & Fun
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the currency of Japan?", "The currency of Japan is the Japanese Yen."),
    ("What is the currency of the United States?", "The currency of the United States is the US Dollar."),
    ("What do bees produce?", "Bees produce honey."),
    ("How many days are in a leap year?", "A leap year has 366 days, compared to 365 days in a normal year."),
    ("How many months are in a year?", "There are 12 months in a year."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("What colors are in a rainbow?", "The main colors of a rainbow are red, orange, yellow, green, blue, indigo, and violet."),
]

# Variations to augment training stability
QUESTION_TEMPLATES = [
    "User: {}\nAssistant: {}\n<|end|>\n",
    "User: {}\nAssistant: {}\n<|end|>\n",
]

def main():
    random.seed(42)
    output_path = "qa_data.txt"

    lines = []
    # Repeat and shuffle pairs to give the network adequate training iterations
    for _ in range(80):
        shuffled_pairs = QA_PAIRS.copy()
        random.shuffle(shuffled_pairs)
        for q, a in shuffled_pairs:
            template = random.choice(QUESTION_TEMPLATES)
            lines.append(template.format(q, a))

    dataset_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dataset_text)

    print(f"[✓] Generated {output_path}")
    print(f"[*] Total characters: {len(dataset_text):,}")
    print(f"[*] Total QA examples: {len(QA_PAIRS) * 80:,}")

if __name__ == "__main__":
    main()
