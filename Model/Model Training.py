#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)
from sklearn.utils import class_weight
from sklearn.preprocessing import label_binarize
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping

#Data loading
with open("new_features_entity1_entity2.pkl", "rb") as f_feat:
    new_feature = pickle.load(f_feat)

with open("new_labels_entity1_entity2.pkl", "rb") as f_label:
    new_label = pickle.load(f_label)


#Definition of the initial model
def create_model(input_shape, output_dim, is_multiclass):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    
    if is_multiclass:
        model.add(Dense(output_dim, activation='softmax'))
        optimizer = optimizers.Adam(learning_rate=1e-2)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        optimizer = optimizers.Adam(learning_rate=1e-2)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


#The final integration model
def create_final_model(input_shape, output_dim, is_multiclass):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    
    if is_multiclass:
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# Initial settings
kf = KFold(n_splits=10, shuffle=True, random_state=42)

ACC, AUROC, AUPRC = [], [], []
Precision, Recall, F_score = [], [], []
TN, FP, FN, TP = [], [], [], []

 
is_multiclass = False  
output_dim = 1         


fold_index = 0
for train_index, test_index in kf.split(new_feature):
    print(f"Processing fold {fold_index}...")
    
    X_train, X_test = new_feature[train_index], new_feature[test_index]
    y_train, y_test = new_label[train_index], new_label[test_index]
    
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.flatten()
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    
    models = []
    loss_history, acc_history = [], []
    
    for j in range(X_train.shape[1]):
        combinations = X_train[:, j, :].reshape(X_train.shape[0], X_train.shape[2])
        
        if is_multiclass:
            class_id = list(range(output_dim))
            labels = label_binarize(y_train[:, j], classes=class_id)
        else:
            labels = y_train.reshape(len(y_train), 1)
        
        model = create_model(combinations.shape[1], output_dim, is_multiclass)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            combinations, labels,
            epochs=50, batch_size=32,
            callbacks=[early_stopping],
            class_weight=class_weights_dict,
            verbose=0
        )
        
        models.append(model)
        loss_history.append(history.history['loss'][-1])
        acc_history.append(history.history['accuracy'][-1])
    
    # Preparing for PSO
    positions = [model.get_weights() for model in models]
    personal_best_positions = positions.copy()
    personal_best_losses = loss_history.copy()
    
    global_best_idx = np.argmin(personal_best_losses)
    global_best_position = personal_best_positions[global_best_idx]
    global_best_loss = personal_best_losses[global_best_idx]
    
    # PSO parameters
    bounds = [-0.01, 0.01]
    num_particles = len(positions)
    velocities = [np.random.uniform(bounds[0], bounds[1], w.shape) for w in positions[0]]
    velocities = [velocities for _ in range(num_particles)]  # یکسان برای همه ذرات
    
    w_inertia = 0.7
    c1 = 1.5
    c2 = 1.5
    
    num_iterations = 10
    for iteration in range(num_iterations):
        print(f"PSO iteration {iteration}...")
        
        for i in range(num_particles):
            #Update Velocity
            r1 = np.random.rand()
            r2 = np.random.rand()
            new_velocity = []
            new_position = []
            
            for v, p_best, g_best, pos in zip(
                velocities[i],
                personal_best_positions[i],
                global_best_position,
                positions[i]
            ):
                vel = (w_inertia * v) + (c1 * r1 * (p_best - pos)) + (c2 * r2 * (g_best - pos))
                pos_new = pos + vel
                new_velocity.append(vel)
                new_position.append(pos_new)
            
            velocities[i] = new_velocity
            positions[i] = new_position
            
            # Update Weights
            models[i].set_weights(new_position)
            
            # Preparing data for retraining
            combinations = X_train[:, i, :].reshape(X_train.shape[0], X_train.shape[2])
            if is_multiclass:
                class_id = list(range(output_dim))
                labels = label_binarize(y_train[:, i], classes=class_id)
                models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                labels = y_train.reshape(len(y_train), 1)
                models[i].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = models[i].fit(
                combinations, labels,
                epochs=50, batch_size=32,
                callbacks=[early_stopping],
                class_weight=class_weights_dict,
                verbose=0
            )
            
            # New loss assessment
            new_loss = history.history['loss'][-1]
            
            
            if new_loss < personal_best_losses[i]:
                personal_best_losses[i] = new_loss
                personal_best_positions[i] = new_position
            
            
            if new_loss < global_best_loss:
                global_best_loss = new_loss
                global_best_position = new_position
    
    # Merging the results of the models for final training
    train_outputs = []
    train_labels = []
    test_outputs = []
    test_labels = []
    
    for i in range(num_particles):
        # Training outputs
        train_comb = X_train[:, i, :].reshape(X_train.shape[0], X_train.shape[2])
        if is_multiclass:
            class_id = list(range(output_dim))
            train_lab = label_binarize(y_train[:, i], classes=class_id)
        else:
            train_lab = y_train.reshape(len(y_train), 1)
        
        train_labels.append(train_lab)
        train_outputs.append(models[i].predict(train_comb))
        
        # Test outputs
        test_comb = X_test[:, i, :].reshape(X_test.shape[0], X_test.shape[2])
        if is_multiclass:
            class_id = list(range(output_dim))
            test_lab = label_binarize(y_test[:, i], classes=class_id)
        else:
            test_lab = y_test.reshape(len(y_test), 1)
        
        test_labels.append(test_lab)
        test_outputs.append(models[i].predict(test_comb))
    
    # Data preparation for the final integration model
    def prepare_integrate_data(output_list, label_list):
        integrated_X = []
        integrated_y = []
        for idx in range(len(output_list[0])):
            temp_out = []
            temp_lab = []
            for p in range(num_particles):
                temp_out.append(output_list[p][idx])
                temp_lab.append(label_list[p][idx])
            integrated_X.append(np.array(temp_out))
            integrated_y.append(np.array(temp_lab[0]))
        return np.squeeze(np.array(integrated_X)), np.array(integrated_y)
    
    train_lst, train_label_lst = prepare_integrate_data(train_outputs, train_labels)
    test_lst, test_label_lst = prepare_integrate_data(test_outputs, test_labels)
    
    # Traing the final model
    final_model = create_final_model(train_lst.shape[1], output_dim, is_multiclass)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    final_model.fit(
        train_lst, train_label_lst,
        epochs=50, batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Predict and final assessment
    predicted_values = final_model.predict(test_lst)
    if is_multiclass:
        max_indices = np.argmax(predicted_values, axis=1)
        binary_pred = np.zeros_like(predicted_values)
        binary_pred[np.arange(len(predicted_values)), max_indices] = 1
    else:
        binary_pred = (predicted_values > 0.5).astype(int)
    
    ACC.append(accuracy_score(test_label_lst, binary_pred))
    AUROC.append(roc_auc_score(test_label_lst, predicted_values))
    AUPRC.append(average_precision_score(test_label_lst, predicted_values))
    Precision.append(precision_score(test_label_lst, binary_pred))
    Recall.append(recall_score(test_label_lst, binary_pred))
    F_score.append(f1_score(test_label_lst, binary_pred))
    tn, fp, fn, tp = confusion_matrix(test_label_lst, binary_pred).ravel()
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)

    fold_index += 1

