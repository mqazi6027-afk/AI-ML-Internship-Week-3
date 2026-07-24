
print("=" * 65)
print("      M-Tech AI/ML Internship - Experiment 11")
print(" Natural Language Processing (NLP)")
print("=" * 65)

text = input("\nEnter a sentence: ")

lower = text.lower()

words = lower.split()

word_count = len(words)

char_count = len(text)

unique_words = set(words)

print("\nProcessed Text")
print("----------------------------")
print("Lowercase :", lower)
print("Words :", words)
print("Total Words :", word_count)
print("Characters :", char_count)
print("Unique Words :", len(unique_words))

print("\nExperiment 11 Completed Successfully")
print("=" * 65)