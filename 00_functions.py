import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, RocCurveDisplay)

def top_words_subreddit(df, bysubreddit=True, ngram=(1,1), stopwords=None, i=50, max_features=500): 

    '''
    This function takes a DataFrame and outputs the top i (default 50) appearing words 
    in each subreddit. It utilizes the CountVectorizer from sklearn.feature_extraction.text 
    to analyze the 'combined_text' column and extracts the top appearing words as a DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least the columns 'combined_text' and 'is_stoicism'.
        bysubreddit (bool, optional): A boolean flag indicating whether to analyze words by subreddit. Default is False.
        ngram_range (tuple, optional): An optional hyperparameter specifying the n-gram range for CountVectorizer, 
                                        e.g., (1, 1) for unigrams or (1, 2) for unigrams and bigrams. Default is (1, 1).
        stopwords (list, optional): An optional list of stopwords to be used with CountVectorizer. Default is None.
        i (int, optional): The quantity of top words to include for each subreddit. Default is 50.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'buddhism' and 'stoicism', 
                        containing counts of the top appearing words, where words are the index.
    '''


    cvec = CountVectorizer(ngram_range=ngram, stop_words=stopwords, max_features=max_features)
    df_token = cvec.fit_transform(df['combined_text'])

    # Create your count matrix/dataframe using get_feature_names()
    c_df = pd.DataFrame(df_token.todense(), columns=cvec.get_feature_names_out())

    if bysubreddit == True:
        # Join df['is_stoicism'] CountVectorized df using the index values as key
        c_df = c_df.join(df['is_stoicism'])
    
        # Subset this result by class so you have 2 count matrices
        bud_df = c_df[c_df['is_stoicism'] == 0].copy()
        sto_df = c_df[c_df['is_stoicism'] == 1].copy()

        # Calculate sums and sort for both classes
        bud_sum_sorted = bud_df.drop('is_stoicism', axis=1).sum().sort_values(ascending=False).head(i)
        sto_sum_sorted = sto_df.drop('is_stoicism', axis=1).sum().sort_values(ascending=False).head(i)

        # Create DataFrames for top words
        bud_df_top = pd.DataFrame(bud_sum_sorted, columns=['buddhism'])
        sto_df_top = pd.DataFrame(sto_sum_sorted, columns=['stoicism'])

        # Perform inner and outer joins
        combined_df_inner = bud_df_top.join(sto_df_top, how='inner')
        combined_df_outer = bud_df_top.join(sto_df_top, how='outer')

        # Concatenate and return the final output
        combined_df = pd.concat([combined_df_inner, combined_df_outer])

        return combined_df
    
    else: 
        return c_df.sum().sort_values(ascending=False).head(i)
    


def gs_eval(X_train, y_train, X_test, y_test, model_name):

    """
    Evaluates a model which has used GrideSearchCV

    This function takes training and testing data along with a GridSearchCV model,
    prints the best parameters and scores obtained from the GridSearchCV,
    fits the best estimator on the training data, and evaluates its performance
    on both training and testing sets. It also makes predictions on the test set.

    Parameters:
    X_train (array-like): Training feature dataset.
    y_train (array-like): Training target dataset.
    X_test (array-like): Testing feature dataset.
    y_test (array-like): Testing target dataset.
    grid_search_model (GridSearchCV object): Pre-defined GridSearchCV model with specific parameters and estimator.

    Returns:
    tuple: A tuple containing the best model from GridSearchCV and the predictions made on the test set.
    """

    print("Best parameters:", model_name.best_params_)
    print("Best score:", model_name.best_score_)

    best_model = model_name.best_estimator_
    best_model.fit(X_train, y_train)
    print("Train score: ", best_model.score(X_train, y_train))
    print("Test score: ", best_model.score(X_test, y_test))

    preds = best_model.predict(X_test)

    return best_model, preds