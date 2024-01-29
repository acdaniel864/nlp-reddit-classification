# Data Science NLP Project - Classifying Philosophical Discourse
**An Exploration of Buddhism and Stoicism Subreddit Posts**

## Problem Statement
Many similarities have been drawn between the teachings of Buddhism and Stoicism. Given this, can the discourse between the two be accurately classified (with 90% or more accuracy) on unseen data? If so, does this hold even as technical terminology: like philosophers' names and writings are excluded from the model?

## Introduction
Two philosophies delicately crafted three decades and 10,000 km apart. In a time long before microchips; when the best computer humans had access to was to close their eyes and think.

One philosophy stemming from ancient Greece, the other ancient India — both standing the test of time and both coming to surprisingly similar conclusions, albeit perhaps, with different vocabulary.

> “The mind is everything. What you think you become.” – Buddha  
> “The happiness of your life depends upon the quality of your thoughts.” – Marcus Aurelius

> “Pain is certain, suffering is optional.” – Buddha  
> “The wise man accepts his pain, endures it, but does not add to it.” – Marcus Aurelius

In this Natural Language Processing (NLP) project, over 3000 distinct Reddit submissions are scraped from two subreddits: r/stoicism and r/buddhism via Python Reddit API Wrapper (PRAW). A machine learning classification model is then built, designed to distinguish between the subreddits based only on the text within scraped submissions. Finally the models used are analysed and some models with restricted input features are evaluated. Through this process it is hoped that some revelations are made as to the similarities and differences in discourse between the philosophical movements. 

## Project Aims
1. Build a high-performing classification model that can distinguish between the two subreddits with at least 90% accuracy on unseen data.
2. Analyze how the removal of technical terminology—philosophers' names, books, writings, and specific technical terminology—affects model performance.
3. Reveal other insights about the similarities or differences in discourse between the two subreddit communities.

## Similarities
The ultimate goal in Buddhism is liberation from suffering through attaining enlightenment (Nirvana), while for stoics its living virtuously with internal freedom independent from external events (Eudaimonia). To touch briefly on other similarites between Buddhism and Stoicism. Both philosophies: 
- Advocate for introspection, observation, and meditation.
- Exalt the present moment as the path to contentment and truth.
- Emphasize the impermanence of all things.
- Offer practical, secular philosophies for living a better life.
- Focus on inner peace, self-control, and personal responsibility.
- Utilise meditation as a tool for reflection and growth.
- View the universe as interconnected and teach that the universe is one. 
- Teach to embrace death: 
    - “Buddhism teaches that.. death is inevitably bound up in life… not to be feared or avoided. “
    - “Memento mori is the term used by the Stoic philosophers to remind us that death is inevitable and that our ability to hold that truth in mind helps us to live better and more fully.”
    Source of quotes [Psychology Today's article](https://www.psychologytoday.com/intl/blog/buddhist-psychology-east-meets-west/202207/the-happy-overlap-between-stoicism-and-buddhism).

We see some very similar foundational ideas stemming from two completely different languages and cultures. However, through centuries of interpretation and translation to arrive at the present day, to what extent has the discourse between the two philosophies converged? This is the broader - more interesting - question that the project will contribute towards. Though an complete objective answer to this is out of the scope of this one NLP binary classification project.  

## Process Overview

### 1. Collecting Data
- Data was scraped from Jan 18th to Jan 26th, 2024, using PRAW via the Reddit API.
- Based on experience a scrape every 2 days was sufficient to ensure no new posts were missed, however more were done in attempts to scrape more data from the past.
- Unable to scrape more than 1000 posts into the past from Jan 18th 2024 due to Reddit API restrictions. 
- 11 scrapes completed in total per subreddit.
- 6336 distinct posts were scraped. 

### 2. Cleaning and Exploratory Data Analysis (EDA)
- Removed duplicate values from dataframes.
- Combined dataframes into one.
- Noted a high proportion of posts with no self-text; merged self-text and title columns.
- Created the numerical target variable.
- Removed punctuation from text.
- Created a word count column per post to find average word count: Buddhism posts - 100 words; Stoicism posts - 143 words.
- Total word count: Buddhism - 327,513; Stoicism - 436,787.
- Plotted word count histograms by subreddit.
- Created 'top_words_subreddit' function (see functions.py file).
- Initial list of frequent words suggests removing stop words may help if overfitting occurs.
- Frequent shared words indicate potential use of TF-IDF in preprocessing may be benefical to increase weight on less common more distinct words.
- Checked top-appearing brigrams - many of which were pronouns prepositions and conjunctions (typical stop words).
- Checked top-appearing trigrams, revealing significant URL presence.
- Approximately 6% of submissions contained URLs; removed these from the corpus.
- Approximately 3% of documents contained emojis; removed emojis.
- Identified a baseline score of 51% for non-stoicism posts.
- Exported cleaned dataset 'stoicism_buddhism_clean.csv".

### 3. Modeling the Data
- Created initial simple logistic regression model - for coefficient interpretation and for use as baseline model. 
   - Initial model showed high accuracy at 0.9 cross-validation score. 
- Coefficient Analysis:
   - Suggested stemming might be useful for better generalization (and reduced computational costs).
   - Stoicism-related words among the top 50 largest coefficients.
   - Interesting top 50 coefficient words: control, quote, philosophy, man, hit, wisdom, training, virtue, fate, indifference, belief, care, practices, thankful, grumpy, artists, political and working. 
   - Many Buddhism-related words had the lowest coefficients.
   -Interesting bottom 50 coefficient words: practice, beautiful, letting, peaceful, peace, hurt, painting, statue, try, impermanent.
- Combination of CVEC pre-processing and TF-IDF tested; TF-IDF yielded lowest bias and variance.
- Explored effects of stemming and stop word inclusion.
- Focused on two models: Logistic Regression and Random Forests.
- Defined optimal production model:   
    - TD-IFD pre-processing with Logistic Regression and 0.918 cross validation score.

### 4. Evaluating and Restricting the Models
- **Production Model - Logistic Regression Model:**
   - Balanced Accuracy: 0.92; F1 score: 0.92.
   - Specificity higher than sensitivity, meaning the model is better at classifying the buddhist subreddits (the negative class) correctly (95% of the time) and slightly worse at predicting the stoicism submissions (the positive class) correctly (89% of the time)
   - Isolated 95 erroneous rows to see if any conclusions could be drawn. Nothing revealing seen in the top appearing words: like, life , people, just, suffering. 
   - ROC curve indicates high model separability (AUC: 0.98).
- **Restricting Words in the Model:**
   - Removed technical jargon and specific terminology. With a custom list of 261 identifying words perviously seen to be extreme (highest/lowest) coefficients in logreg model. Focused on: philosophers names, books, writings and specific terminology/phases from Stoicism and Buddhism based on domain knowledge.
   - Accuracy fell by around 10%.
   - Model still accurate at 82% with technical jargon removed. 
   - Proportional changes in recall and precision, suggests that the model relied fairly equally on technical Jargon to classify submissions between the two subreddits. 
   - Then attempted removal of 137 words related to rationality, logic, and emotions but yielded no conclusive results. Model remained fairly stable suggesting these words did not have any speficic distinguishing power for the model.
- **Model Comparison:**
   - Random Forest classifier showed decreased balanced accuracy compared to Logistic Regression.
   - Random Forest had more balanced recall and precision however, Random Forest was underperforming the Logistic Regression in identifying the positive class (r/stoicism). 
   - Random Forest exhibited higher variance and bias on the test dataset.
   - Random Forest showed a lower F1 score compared to logistic regression.


### 5. Conclusion and Summary of Results:
- **High Accuracy**: The model achieved 92% accuracy on unseen data, demonstrating its robustness and predictive power - exceeding our target in problem statement of over 90%.
- **Extreme Coefficient Technical Terms:**
    - Many of the terms with highest or lowest coefficients (representing the log odds of a post being in r/stoicism if that word is present in the submission) were terms like 'buddhism', 'buddhist', 'stoic', 'stoicism' and philosophers names 'marcus' 'aurelius' or technical vocab 'karma', 'dharma'.
    - For example, A post containing the word 'epicutus' increased log odds of the post being from r/stoicism by 4.1. 
    - A post containing the word 'karma' decreased log odds of the post being from r/stoicism by 3.4. 
- **Technical Jargon Defined**: The model's performance was significantly influenced by technical jargon. This jargon includes names of philosophers, philosophical writings, and words in Greek, Latin, Pali, or Sanskrit.
- **Decrease in Accuracy When restricted**: When technical jargon was removed, both test accuracy and balanced accuracy dropped by 11%.
- **Increase in Errors**: The removal of technical jargon resulted in a 126% increase in both false negatives and false positives, indicating the model's reliance on this jargon for classification.
- **Above Baseline Performance**: Even with reduced accuracy (81%), the model performed 30% above the baseline, effectively classifying submissions. Which suggests despite having very similar foundational ideas, being translated to english and being re-intrepretted over centuries, the philosophies appear to have remarkably distinct vocabularies:
    1. **Stoicism vs Buddhism**: 
        - Stoicism subreddit often involves terms like 'virtues' and 'virtuous', whereas Buddhism focuses on 'enlightenment'. A submission containing the word 'virtue' increases log odds of a post being from r/stoicism by 1.7, whereas a submission containing the word 'enlightenment' decreases log odds of a post being from r/stoicism by 2.1.
        - Stoics frequently mention 'care', while Buddhists emphasize 'compassion' - these were distribguidhing coefficients. 
    2. **Vocabulary Differences in bottom/top 50 coefficients**:
        - Stoicism subreddit had more verbs such as: 'control', 'live', 'react', 'work', 'gonna', 'respond', 'act', 'deal', 'think', 'going', 'handle', 'stop', 'advice'.
        - Buddhism subreddit verbs were: 'thank', 'practice', 'suffering', 'teaching', 'painting', 'killing'.
    3. **Emotional and Mental Statesin bottom/top 50 coefficients**:
        - Buddhism subreddit terms related to mental states and emotions appear more prevalent: 'suffering', 'peace', 'compassion', 'emptiness', 'mindful', 'enlightenment', 'enlightened', 'beautiful'.
        - Stoicism subreddit: 'emotions', 'emotionally', 'emotion', 'react', 'strength'.


### Data Dictionary

| Column Name | Data Type | File | Description |
|---|---|---|---|
| title | object| stoicism_buddhism_clean.csv | The title of the submission. |
| selftext| object | stoicism_buddhism_clean.csv | The submissions’ selftext - an empty string if a link post|
| subreddit| object| stoicism_buddhism_clean.csv | Provides an instance of Subreddit |
| created_utc | object| stoicism_buddhism_clean.csv| Time the submission was created, represented in Unix Time.|
| name        | object| stoicism_buddhism_clean.csv| Fullname of the submission - a unique ID name used in Reddit backend |
| upvote ratio| float | stoicism_buddhism_clean.csv |The percentage of upvotes from all votes on the submission, as float from 0-1|
| num_upvotes	| int| stoicism_buddhism_clean.csv |Absolute number of upvotes receieved on reddit submission|
| combined_text	| object| stoicism_buddhism_clean.csv |Text from title and selftext combined - feature variable|
| is_stoicism	| boolean | stoicism_buddhism_clean.csv | Binary column, 1 if post is from stoicism subreddit, 0 if buddhism - target variable|
| 	contains_https		| boolean | stoicism_buddhism_clean.csv | Binary 1 if post contains URL, 0 if not|
| 	contains_emoji | boolean | stoicism_buddhism_clean.csv | Binary 1 if post contains emoji, 0 if not|



### References

##### Philosophical Comparisons
- [Stoicism and Buddhism: A Comparison](https://philosophyasawayoflife.medium.com/stoicism-and-buddhism-a-comparison-58d3e86587b)
- [The Stoic Sage - Stoicism and Buddhism](https://thestoicsage.com/stoicism-and-buddhism/)
- [Classical Wisdom - Stoicism and Buddhism: Two Sides of the Same Coin](https://classicalwisdom.com/philosophy/stoicism-and-buddhism/)
- [Psychology Today - The Happy Overlap Between Stoicism and Buddhism](https://www.psychologytoday.com/intl/blog/buddhist-psychology-east-meets-west/202207/the-happy-overlap-between-stoicism-and-buddhism)
- [Quora Discussion: Why isn't Stoicism considered a religion?](https://www.quora.com/Why-isnt-Stoicism-considered-a-religion-since-Epictetus-mentions-God-so-much-in-his-Discourses)
- [Daily Stoic - Stoicism and Buddhism: How Similar Are They?](https://dailystoic.com/stoicism-buddhism/)
- [Rationally Speaking - Buddhism, Epicureanism, and Stoicism](https://rationallyspeaking.blogspot.com/2013/02/buddhism-epicureanism-and-stoicism.html)

##### Technical and Programming References
- [FreeCodeCamp - Regular Expression for a URL](https://www.freecodecamp.org/news/how-to-write-a-regular-expression-for-a-url/)
- [Stack Overflow - Adding Stemming Support to CountVectorizer (scikit-learn)](https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn)
- [Stack Overflow - Best Regular Expression for Valid URL](https://stackoverflow.com/questions/161738/what-is-the-best-regular-expression-to-check-if-a-string-is-a-valid-url?page=2&tab=votes)
- [GitHub Gist - Regular Expression for Fast Punctuation Removal](https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b)
- [IncludeHelp - Fast Punctuation Removal with Pandas](https://www.includehelp.com/python/fast-punctuation-removal-with-pandas.aspx)
- [Stack Overflow - Adding Words to NLTK Stoplist](https://stackoverflow.com/questions/5511708/adding-words-to-nltk-stoplist)
- [Towards Data Science - Visualizing a Decision Tree from a Random Forest in Python Using Scikit-Learn](https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c)
- [Towards Data Science - Decision Trees Explained: Entropy, Information Gain, Gini Index, CCP Pruning](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c)

