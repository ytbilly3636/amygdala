# amygdala
amygdala model

### What is amygdala model
Amygdala is an area of brain.
The amygdala causes fear conditioning; a kind of classical conditioning.
Amygdala model is inspired by the amygdala and can be used for classical conditioning.

### Requirement
* Python
* NumPy
* OpenCV (Optional)

### Quick Start
```
$ python sonoh.py
```

### How to scrach amygdala model
First of all, import layers of amygdala model.
In the amygdala, there are parts called lateral nucleus of amygdala (LA) and central nucleus of amygdala (CE).
```
from layer import LateralNucleus as LA
from layer import CentralNucleus as CE
```

Secondly, construct a class of amygdala using LA and CE.
Multiple self-organizing maps (SOMs) are used as models of LA.
So, specify the number of SOMs and its parameters.
```
class Amygdala(object):
    def __init__(self):
        # la_size: the number of SOMs in LA
        # la_map_size: the number of neurons in each SOM
        # ls_in_size: the dimension of sensory stimulus
        self.la = LA(la_size=1, la_map_size=(8, 8), la_in_size=3)
        
        # in_size: la_size * la_map_size[0] * la_map_size[1]
        # out_size: the dimension of emotional stimulus
        self.ce = CE(in_size=1*8*8, out_size=2)
```

Sensory stimulus are input into an LA.
The LA integrates and classifies the stimulus.
Classified stimulus are input into an CE.
The CE outputs emotional response.
So, add a following function to the class of amygdala.
```
    def inference(self, xs, var=0.4):
        # xs: sensory stimulus
        # var: hyper parameter of outputs of LA
        
        h = self.la.inference(xs, var)
        y = self.ce.inference(h)
        return y
```

The SOMs in LA obtain self-organized map from sensory stimulus.
The CE conditions the sensory stimulus and emotional stimulus.
There learning are operated by a following function.
```
    def update(self, t, lr_la=0.01, var_la=0.5, lr_ce=0.1):
        # t: emotional stimulus, typically one-hot vector
        # lr_la: learning rate of LA
        # var_la: learning hyper parameter of LA
        # lr_ce: learning rate of CE
        
        self.la.update(lr_la, var_la)
        self.ce.update(t, lr_ce)
```

By using added functions, classical conditioning can be done.
```
# instance of amygdala model
amy = Amygdala()

# x: sensory stimulus must be a list of numpy array, and the shape is (batch, in_size).
# t: emotional stimulus should be a one-hot vector.
x = [np.array([[0.0, 1.0, 0.0]])]
t = np.eye(2)[0].reshape(1, -1)

# classical conditioning
y = amy.inference(x)
amy.update(t)
```

In order to make training effective, pretrainig of LA is recommended.
In this case, random input vector is used for pretraining.
```
def pretraining():
    dammy_t = np.zeros((1, 2))

    for i in six.moves.range(1000):
        x = [np.random.rand(1, 3)]
        amy.inference(x)
        amy.update(dammy_t, lr_la=0.1, var_la=1.0, lr_ce=0.0)
```

---

scripted by

Yuichiro Tanaka
tanaka.yuichiro483@mail.kyutech.jp
Tamukoh Laboratory, Kyushu Institute of Technology
