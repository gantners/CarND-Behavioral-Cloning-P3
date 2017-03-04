import pandas as pd
import my_model
import visual


def eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape((y_pred.shape[0],))

    size = 200 if len(X_test) > 200 else len(X_test)

    ax = pd.DataFrame({'predicted': y_pred[:size], 'actual': y_test[:size]}).plot()
    ax.set_ylabel("steering angle")


def eval_visual(model, input_img):
    visual.visualize_output_layer(model, input_img)
