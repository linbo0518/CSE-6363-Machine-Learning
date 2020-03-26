from naive_bayes import NaiveBayes

model = NaiveBayes()
model.fit('vertebrate_20.txt',
          target_col="ClassLabel",
          ignore_col="Name",
          start_idx=0,
          end_idx=10)
print("Likelihood:")
model.likelihood()
print("\nPosterior Probability:")
model.predict('vertebrate_20.txt', start_idx=10, end_idx=21, verbose=True)
