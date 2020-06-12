# Hackaton_iml
a model for git repository classification was trained using TFIDF vectorization and linear SVM classification.'
the class GitHubClassifier load the already trained model and predict a new sample (in the form of list of strings) regarding to 7 possible git repositories (labels). code of the selection and train process available inside.
method description:
TFIDF word embeding is a methos to vectorize words in text using TF (term frequency) IDF (inverse document frequency) scoring. the TF term measure the frequency of a word (or a n-gram object) in the text and the IDF term give weight to words that occur more rarely in the text. this way we get a sparse vector with the non zero elements as the uniqe words in the text.
Linear SVM - a linear halfspace classifier that find seperator using max-margin method. the regularizetion term is in the form of the number of expections samples (inside the margin) that allowed
our model get a list of string and convert is to design matrix using TfidfVectorized and them classify it using LinearSVM. both packages are implemented with sklearn.
