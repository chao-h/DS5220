import numpy as np

class Softmax:
    #Convert category
    def yconvert(self, y,num):
        Y = np.zeros((len(y),num))
        for i in range(len(y)):
            Y[i,y[i]] = 1
        return np.matrix(Y)

    #Softmax function
    def softmax(self, x):
        sm = (np.exp(x)/np.sum(np.exp(x),axis = 1))
        return np.matrix(sm)

    #Cross-entropy loss fucntion
    def cross_entropy(self, prob, y, lam,w):
        loss = -np.sum(np.multiply(np.log(prob), y)) +  lam*np.sum(abs(w))
        return loss

    #Batch gradient descent
    def batch_gradient(self, x, y, prob, lam, w):
        grad = -np.dot(x.T, (y - prob)) + lam*abs(w)
        return grad

    #Mini Batch gradient descent
    def mini_batch(self, x, y, prob, lam, w, i, j):
        grad = -np.dot(x[i:j,].T, (y[i:j,] - prob[i:j,])) + lam*abs(w)
        return grad

    #prediction
    def prediction(self, x,w):
        probs = self.softmax(np.dot(x,w))
        preds = np.argmax(probs, axis = 1)
        return preds

    #correct rate
    def rate(self, x,w,y):
        pred = self.prediction(x,w)
        return np.sum(pred.T == y)/len(y)

    #Main caculation function
    def fit_predict(self, xtrain, ytrain, xvalid, yvalid, lam, alpha, e, opt = 0):
        w = np.matrix(np.zeros((xtrain.shape[1],5)))
        loss = 0
        r = []
        n = xtrain.shape[0]
        y = self.yconvert(ytrain, 5)
        j = 0
        size = 4000
        for i in range(1000):
            
            prob = self.softmax(xtrain * w)
                
            loss0 = loss
            
            loss = (1/n) * self.cross_entropy(prob, y, lam, w)
            
            grad = (1/n) * self.mini_batch(xtrain, y, prob, lam, w, j, j + size)
            
            j = (j + size)%(n-size)
            
            w = w - (alpha * grad) 
            #print((j,j+size))
            #print(abs(loss0-loss),e)
            if opt == 2 or 3:
                r.append(self.rate(xvalid, w, yvalid))
            if (abs(loss0-loss) < e):
                break 
        if opt == 0:
            return self.rate(xvalid, w, yvalid)
        elif opt == 1:
            return self.rate(xvalid, w, yvalid), self.prediction(xvalid, w)
        elif opt == 2:
            return self.rate(xvalid, w, yvalid), r
        elif opt == 3:
            return w
        else:
            return self.rate(xvalid, w, yvalid), self.prediction(xvalid, w), r   