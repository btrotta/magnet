## MagNet Competition

Code for the second-place winner of the MagNet competition to predict disturbances in the Earth's magnetic field.
The competition is sponsored by National Oceanic and Atmospheric Administration (NOAA). 
See the competition website for more details: https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/

The model is a fairly small convolutional neural network with minimal preprocessing of the data. 
It trains in around 15 minutes on CPU.


#### Model

The model is a convolutional neural network with rectified linear activations. It uses tensorflow with the keras framework. 

The model consists of a set of layers which apply convolutions to detect patterns at progressively longer time spans. 
At each convolutional 
layer (except the last), a convolution is applied having size 6 with stride 3, which reduces the size of the output data 
relative to the input. (The last convolution has small input size, so it just convolves all the 9 input points 
together.) Thus the earlier layers recognize lower-level features on short time-spans, and these are aggregated into higher-level
patterns spanning longer time ranges in the later layers. Cropping is applied at each layer which removes a few data points at the beginning at the 
sequence to ensure the result will be exactly divisible by 6, so that the last application of the convolutional filter will 
capture the data at the very end of the sequence, which is more important to the prediction. 

After all the convolutional 
layers is a layer which concatenates the last data point of each of the convolution outputs (achieved by left-cropping 
all but the last point). This concatenation is then fed these into a dense layer. The idea of taking the last data point of each convolution 
is that it should give a representation of the features  at different timespans leading up to the prediction time. 
For example, the last data point 
of the first layer gives the features of the hour before the prediction time, then the second layer gives 
the last 6 hours, etc.

#### Features used and preprocessing

I used only the solar wind and sunspots data; I found that the satellite data didn't help my model. 
I used the following columns: `bt`, `density`, `speed`, `bx`, `by`, `bz`. 
The features `temperature` and `source` did not seem to improve the model. I used the data in the 
`(x, y, z)` coordinate system, rather than the angular co-ordinate systems, because it would be harder to calculate 
mean and standard deviations for angle data in degrees.

I excluded from training periods where the temperature is `< 1`. Missing data is filled 
by linear interpolation (to make the result less noisy, we interpolate using a smoothed rolling average, 
rather than just the 2 points immediately
before and after the missing part).

Data is normalized by subtracting the median and dividing by the inter-quartile range (I used this approach rather 
than the more usual mean and standard deviation because some of the variables have asymmetric distributions with 
long tails).

I aggregated the training data in 10-minute increments, taking the mean and standard deviation of each feature in 
the increment. This could have alternatively been done as the first layer of the neural network, but it's faster 
to do it as a preprocessing step.


#### Ensembling

I trained an ensemble of 5 models, by excluding 20% of the training data each time. This was done by 
 splitting the training data into months and randomly selecting 20% of the months to hold out. I split by months 
rather than hours because subsequent hours are likely to be correlated, meaning test and train sets would be 
more similar, reducing the benefit of the ensemble. The prediction code averages the output of the 5 models.

I trained a separate set of models for time `t` and `t+1`, so there are 10 models in total.

