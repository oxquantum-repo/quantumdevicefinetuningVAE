# Nina's Algorithm

## Contents

* Code structure/graph
* Input requirements and functionality
* Output/ control
* Run
* Function connectivity
* Detailed functionality of each module

---

### Code structure

1. [Optimisation algorithm.ipynb] - File to run code
   1. Calls [factor_vae_bias_sub]
   2. [factor_vae_bias_sub] will call [vae_bias_sub] 
   3. FactorVAE  is the parent class of VAE



```python

```

---

### Input requirements and functionality



```python

```

---

### Output/ control



```python

```

---

### Run

run_random() 

​	Will run algorithm

```python
run_random(savedir, window_small=30, res_low=10, res_high=20, n_walk=30,it=10, window_large_init=30,
               gates_init=[-1150.9539658,-808.79439154,-1815.3845539,-1127.79332466,-1406.53212482,-682.98769082],
               starting_c5_c9_init=[-575,-945],res_save=128)
```

gates_init=[c3, c4, c6, c7, c8, c10],
               starting_c5_c9_init=[c5, c9]

* Loads vae and vae2
  * Only vae2 is used - Based on dataset with translational augmentation
  * c5 and c9 are plungers (x and y axis)

* Will run random()

  * ```python
    for i in range(n_walk):
    ```

    *  Each i in range(n_walk) runs it=60 steps from the starting position

  * in random dv is the step size

    * Pass in as variable
    * Diminishing step size

  * idx_init 

    * 6 options, modifying one of 3 gates by + dv or -dv

  ```python
  idx_init=[np.array([0,-dv]),np.array([0,dv]),np.array([1,-dv]),np.array([1,dv]),np.array([5,-dv]),np.array([5,dv])]
  ```

  - 0,1,5 define the indices of the control gates within the complete gate list defined bygates_init in run_random()

* ```python
  for j in range(it):
  ```

  * Will take it = 60 actions

* Extract new gate voltages

  * ```python
        gates_prop,delta_v, idx,gate_idx, idx_tot, gates_tot,gates=get_new_param(gates,gates_tot, idx,idx_tot)
    ```

    * get_new_param()

      * ```python
        if len(idx)==0:
                if len(idx_tot)==0:
        ```

        * idx is list of actions left
        * of idx_tot == 0 the entire list is exhausted and then just returns to initial point - has not found within branch a more perfect triangle

      * ```python
         else:
        ```

        * random action selected from possible action list

* Low res scan to recenter

* ```python
  5c9_list_prop=blob_detection(data_low,c5c9_list,res_low=res_low,window_large=window_large)
  ```

  	* Returns proposed values for plungers of the centre of triangle
   * If blob detection fails
     	* Check threshold and min_sigma, mas_sigma which are hardcoded into blob detection
        	* Uses blobs (log) not blobs_2, could comment out blobs_2
     	* 

* High res scan to be scored

* ```python
  score_vae,score_vae_trans,score_tot=predict_mod(vae,vae2, data_high)
  ```

  * predict_mod uses vae's trained on 32 x 32 so res_high should always be defined as 32
  * normalise()
    * Normalises current based upon whole plot (from Dom's) - hardcoded
    * If you normalise just in region of triangle then a feint triangle would be scored highly
    * vae is very sensitive to normalisation

* Lower scores are better as they represent the euclidean distance with variance term from targets

* ```python
  idx=[x for x in idx if not np.array_equal(x,np.array([gate_idx,-delta_v]))]
  ```

  * removes reverse of selected action from list of possible actions within branch

* can comment out data_save

---

# To run on server

```python
savedir="//media//oxml//DA12F77512F754CB//Pygor_results//optimization//gamma//" #output directory 

pygor = Pygor.Experiment(mode='jupyter', xmlip="http://129.67.86.107:8000/RPC2",savedir=savedir) # Access pygor where http://129.67.86.107:8000/RPC2 is ip address of Triton 4

run_random(window_small=17, res_low=30, res_high=32, n_walk=3,it=60,gates_init=

[-1356.597083799452, -655.7552004185842, -757.39513943, -1910.53470352, -1315.74489394, -610.0010767257405],  starting_c5_c9_init=[-626, -782] ,savedir=savedir,res_save=128)
```



---

### Function Connectivity



```python

```

---



## Modular Overview

