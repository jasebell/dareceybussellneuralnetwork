(ns darceyneuralnetwork.core
  (:require [clj-ml.io :as wio]
            [clj-ml.data :as data]
            [clj-ml.classifiers :as classifier]
            [clj-ml.utils :as utils]))

(defn load-instance-data [file-url]
  (wio/load-instances :arff file-url))

(defn train-neural-net [training-data-filename class-index opts]
  (let [instances (load-instance-data training-data-filename)
        neuralnet (classifier/make-classifier :neural-network :multilayer-perceptron opts)]
    (data/dataset-set-class instances class-index)
    (classifier/classifier-train neuralnet instances)))

(defn build-classifier [training-data-filename output-filename]
  (let [opts (classifier/make-classifier-options :neural-network :multilayer-perceptron
                                                 {:learning-rate 0.4
                                                  :epochs 10000})
        nnet (train-neural-net training-data-filename 3 opts)]
    (utils/serialize-to-file nnet output-filename)))
