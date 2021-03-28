# Ordinal Logistic Regression
from sklearn.linear_model import LogisticRegression
class OrdinalLogisticRegression():
    def __init__(self,random_state=None,solver='saga',max_iter=500,C=1,penalty='l1',l1_ratio=None,class_weight=None):
        """"
        Define parameters for logistic regressions
        """
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
    
    def fit(self,x,y):
        """
        y should be en encoded target such that set(y) puts it in the expected order
        """
        self.x = x
        self.y = y
        self.targets = list(set(y))
        self.n_targets = len(self.targets)
        self.lr_dict = {}
        running = []
        for target in self.targets[:-1]:
            running.append(target)
            # If y is in the running list of targets we set it to 0.  Otherwise we set y to 1 as shown in the lecture
            # We progressively shift the 'boundary' adding to our running target list each iteration
            # The last iteration is simply all the targets against the last target, hence why we loop through self.targets[:-1]
            y_ = (~np.isin(y,running)).astype(int)
            # Logistic regression fit using the hyperparameters
            self.lr_dict[target] = LogisticRegression(C=self.C,random_state=self.random_state,solver=self.solver,max_iter=self.max_iter,penalty=self.penalty,l1_ratio=self.l1_ratio,class_weight=self.class_weight)
            self.lr_dict[target].fit(self.x,y_)

    def predict(self,x):
        self.pred = {}
        self.ordinal_pred = {}
        for n,target in enumerate(self.targets[:-1]):
            self.pred[target] = self.lr_dict[target].predict_proba(x)
            if n==0:
                self.ordinal_pred[target] = self.pred[target][:,0]
                self.preds = np.array(self.ordinal_pred[n])
            elif (n>0) & (n<self.n_targets-2):
                # Each prediction is the prob that it is class 0 minus the previous prob that it is class 0
                self.ordinal_pred[target] = self.pred[target][:,0] - self.pred[target-1][:,0]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n]))
            elif n==self.n_targets-2:
                self.ordinal_pred[target] = self.pred[target][:,0] - self.pred[target-1][:,0]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n]))
                # The prob of the last target is the prob of the class==1 instead of 0 
                self.ordinal_pred[target+1] = self.pred[target][:,1]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n+1]))
        # Return the highest prob. prediction.  The index returned with argsort is the predicted label
        return self.preds.argsort(axis=0)[-1,:]
    
    def predict_proba(self,x):
        self.pred = {}
        self.ordinal_pred = {}
        for n,target in enumerate(self.targets[:-1]):
            self.pred[target] = self.lr_dict[target].predict_proba(x)
            if n==0:
                self.ordinal_pred[target] = self.pred[target][:,0]
                self.preds = np.array(self.ordinal_pred[n])
            elif (n>0) & (n<self.n_targets-2):
                self.ordinal_pred[target] = self.pred[target][:,0] - self.pred[target-1][:,0]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n]))
            elif n==self.n_targets-2:
                self.ordinal_pred[target] = self.pred[target][:,0] - self.pred[target-1][:,0]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n]))
                self.ordinal_pred[target+1] = self.pred[target][:,1]
                self.preds = np.vstack((self.preds,self.ordinal_pred[n+1]))
        return self.preds.max(axis=0)

    def accuracy(self,x,y):
        preds = predict(x)
        return accuracy_score(y_true=y,y_pred=preds)

    def get_params(self,deep=True):
        if 1==1:
            return {        
                    'random_state':self.random_state,
                    'solver':self.solver,
                    'max_iter':self.max_iter,
                    'C':self.C,
                    'penalty':self.penalty,
                    'l1_ratio':self.l1_ratio
                    }
          
    def set_params(self,**params):
        for k,v in params.items():
            setattr(self, k, v)