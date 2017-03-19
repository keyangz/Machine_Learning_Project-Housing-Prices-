def optimize_and_evaluate_model(pipeline, parameter_grid, method_name, X, Y):
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    gridsearch = GridSearchCV(pipeline, parameter_grid, cv=5, scoring='neg_mean_squared_error')
    gridsearch.fit(X, Y)
    print(gridsearch.best_estimator_)
    print("\nMethod is " + method_name)
    print "Root Mean Square Error", (np.sqrt(abs(gridsearch.best_score_)))
    print "\n#################################################\n"