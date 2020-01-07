import { default as tf } from '@tensorflow/tfjs-node'
import { createModel } from './model.js'
import { train, validation } from './data.js'


const trainDataset = tf.data.generator(train).batch(200).shuffle(200)
const validationDataset = tf.data.generator(validation).batch(13)


createModel().fitDataset(trainDataset, {
  epochs: 20,
  verbose: 1,
  validationData: validationDataset,
  callbacks: tf.node.tensorBoard('../logs', {updateFreq: 'batch'}),
})