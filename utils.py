from keras.models import load_model
from injury import ConnectomeInjury
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from numpy import std
from IPython.display import HTML, display
import tabulate

def generate_synthetic_validation_data(noise):
    injury = ConnectomeInjury() # Generate train/test synthetic data.
    x_valid_y_valid = injury.generate_injury(n_samples=300, noise_weight=noise)
    return x_valid_y_valid


def get_results_from_models(model,noises):
    results = [["noises"]+noises,
               ["mae_alpha"],
               ["stdae_alpha"],
               ["mae_beta"],
               ["stdae_beta"],
               ["r_alpha"],
               ["r_beta"]]
    for i in range(len(noises)):
        noise = noises[i]
        x_valid, y_valid = generate_synthetic_validation_data(noise)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[3], x_valid.shape[2], x_valid.shape[1])
        # load weights into new model
        model.load_weights("Weights/BrainCNNWeights_noise_"+str(noise)+".h5")
        print("Loaded model from disk")
        preds = model.predict(x_valid)
        results[1].append("{0:.2f}".format(100*mae(preds[:,0],y_valid[:,0])))
        results[2].append("{0:.2f}".format(100*std(abs(y_valid[:, 0] - preds[:, 0]))))
        results[3].append("{0:.2f}".format(100*mae(preds[:, 1], y_valid[:, 1])))
        results[4].append("{0:.2f}".format(100*std(abs(y_valid[:,1]-preds[:,1]))))
        results[5].append("{0:.2f}".format(pearsonr(preds[:,0],y_valid[:,0])))
        results[6].append("{0:.2f}".format(pearsonr(preds[:, 1], y_valid[:, 1])))
    display(HTML(tabulate.tabulate(results, tablefmt='html')))