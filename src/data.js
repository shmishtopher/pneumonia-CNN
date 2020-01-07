import { default as tf } from '@tensorflow/tfjs-node'
import { promises as fs } from 'fs'


const healthyTrain = fs.readdir('./data/train/NORMAL', {withFileTypes: true})
const healthyValidate = fs.readdir('./data/val/NORMAL', {withFileTypes: true})
const pneumoniaTrain = fs.readdir('./data/train/PNEUMONIA', {withFileTypes: true})
const pneumoniaValidate = fs.readdir('./data/val/PNEUMONIA', {withFileTypes: true})


export async function* train() {
  for (const [healthy, sick] of zip(...await Promise.all([healthyTrain, pneumoniaTrain]))) {
    const healthyPath = `./data/train/NORMAL/${healthy.name}`
    const sickPath = `./data/train/PNEUMONIA/${sick.name}`

    const healthyFile = fs.readFile(healthyPath)
    const sickFile = fs.readFile(sickPath)

    const healthyTensor = healthyFile.then(buffer => tf.node.decodeJpeg(buffer, 1))
    const sickTensor = sickFile.then(buffer => tf.node.decodeJpeg(buffer, 1))

    yield { xs: tf.image.resizeBilinear(await healthyTensor, [256, 256]), ys: tf.tensor1d([0, 1]) }
    yield { xs: tf.image.resizeBilinear(await sickTensor, [256, 256]), ys: tf.tensor1d([1, 0]) }

    await healthyTensor.then(tensor => tensor.dispose())
    await sickTensor.then(tensor => tensor.dispose())
  }
}


export async function* validation() {
  for (const [healthy, sick] of zip(...await Promise.all([healthyValidate, pneumoniaValidate]))) {
    const healthyPath = `./data/val/NORMAL/${healthy.name}`
    const sickPath = `./data/val/PNEUMONIA/${sick.name}`

    const healthyFile = fs.readFile(healthyPath)
    const sickFile = fs.readFile(sickPath)

    const healthyTensor = healthyFile.then(buffer => tf.node.decodeJpeg(buffer, 1))
    const sickTensor = sickFile.then(buffer => tf.node.decodeJpeg(buffer, 1))

    yield { xs: tf.image.resizeBilinear(await healthyTensor, [256, 256]), ys: tf.tensor1d([0, 1]) }
    yield { xs: tf.image.resizeBilinear(await sickTensor, [256, 256]), ys: tf.tensor1d([1, 0]) }

    await healthyTensor.then(tensor => tensor.dispose())
    await sickTensor.then(tensor => tensor.dispose())
  }
}


function* zip(leftArray, rightArray) {
  const leftIterator = leftArray.values()
  const rightIterator = rightArray.values()

  let leftElement = leftIterator.next()
  let rightElement = rightIterator.next()

  while (!leftElement.done && !rightElement.done) {
    yield [leftElement.value, rightElement.value]
    leftElement = leftIterator.next()
    rightElement = rightIterator.next()
  }
}