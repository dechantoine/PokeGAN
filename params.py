k_size = 5

gen = {"dense": {"pixels":18,
                 "channels":64,
                 "batchnorm":True
                },
       "conv2dts": {"c1": {"channels":64,
                           "kernel_size":k_size,
                           "strides":2,
                           "batchnorm":False,
                           "dropout":False,
                           "dropout_rate":0.5
                          },
                    "c2": {"channels":16,
                           "kernel_size":k_size,
                           "strides":2,
                           "batchnorm":False,
                           "dropout":False,
                           "dropout_rate":0.5
                          },
                    "c3": {"channels":4,
                           "kernel_size":k_size,
                           "strides":2,
                           "batchnorm":False,
                           "dropout":False,
                           "dropout_rate":0.5
                          }
                   },
       "conv2dt_out": {"kernel_size":k_size,
                       "strides":1}
      }

disc = {"conv2d_in": {"channels":128,
                      "kernel_size":k_size,
                      "strides":2,
                      "batchnorm":False,
                      "dropout":True,
                      "dropout_rate":0.2
                },
        "conv2ds": {"c1": {"channels":128,
                           "kernel_size":k_size,
                           "strides":2,
                           "batchnorm":False,
                           "dropout":True,
                           "dropout_rate":0.2
                          },
                    "c2": {"channels":256,
                           "kernel_size":k_size,
                           "strides":2,
                           "batchnorm":False,
                           "dropout":True,
                           "dropout_rate":0.2
                          },
                    "c3": {"channels":256,
                           "kernel_size":k_size,
                           "strides":1,
                           "batchnorm":False,
                           "dropout":False,
                           "dropout_rate":0.5
                          }
                   }
       }