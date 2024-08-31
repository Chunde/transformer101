import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import BilingualDataset
from Modules.TransformerHelper import createTranslationTransformer
from torch.utils.tensorboard import SummaryWriter
import tqdm

sequenceLength = 1000
sourceLang = "en"
targetLang = "zh"
modelSize = 512
batchSize = 128
epochNum = 20
lr = 1e-4


def getAllSentence(dataset, split, lang):
    for item in dataset:
        yield item[split]["translation"][lang]


def getTokenizer(dataset, split, lang):
    tokenizersPath = Path(lang + ".token")
    if not Path.exists(tokenizersPath):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            getAllSentence(dataset, split, lang), trainer=trainer
        )
        tokenizer.save(str(tokenizersPath))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizersPath))


def loadDataset():
    # we download everything, to only download train data
    dataset = load_dataset("opus100", "{sourceLang}-{targetLang}")
    print(dataset)
    trainingDataset = dataset["train"]
    validationDataset = dataset["validation"]
    sourceTrainTokenizer = getTokenizer(dataset, "train", sourceLang)
    targetTrainTokenizer = getTokenizer(dataset, "train", targetLang)

    trainDatasetInput = BilingualDataset(
        trainingDataset,
        sourceTrainTokenizer,
        targetTrainTokenizer,
        sourceLang,
        targetLang,
        sequenceLength,
    )
    validationDatasetInput = BilingualDataset(
        validationDataset,
        sourceTrainTokenizer,
        targetTrainTokenizer,
        sourceLang,
        targetLang,
        sequenceLength,
    )

    maxSourceLen = 0
    maxTargetLen = 0
    for item in trainingDataset:
        sourceIds = sourceTrainTokenizer.encode(item["translation"][sourceLang]).ids
        targetIds = targetTrainTokenizer.encode(item["translation"][targetLang]).ids
        maxSourceLen = max(maxSourceLen, len(sourceIds))
        maxTargetLen = max(maxTargetLen, len(targetIds))

    print(f"max source and target sentence length are: {maxSourceLen}, {maxTargetLen}")

    trainDataLoader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
    validationDataLoader = DataLoader(validationDataset, batch_size=1, shuffle=True)
    
    return trainDataLoader, validationDataLoader, sourceTrainTokenizer, targetTrainTokenizer

def getModel(sourceVocabLen, targetVocabLen):
    model = createTranslationTransformer(
        sourceVocabLen, targetVocabLen, sequenceLength, sequenceLength, modelSize
    )
    return model


def trainModel():
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    Path('model').mkdir(parents=True, exist_ok=True)
    
    trainDataLoader, validationDataLoader, sourceTokenizer, targetTokenizer = loadDataset()
    model = getModel(sourceTokenizer.get_vocab_size(), targetTokenizer.get_vocab_size()).to(device)
    
    #Tensor board
    
    writer = SummaryWriter("TS101")
    
    optimizer = tc.optim.Adam(model.parameter(), lr=lr, eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    loss_function = nn.CrossEntropyLoss(ignore_index=sourceTokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, epochNum):
        model.train()
        
        batchIterator = tqdm(trainDataLoader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batchIterator:
            encoderInput = batch['encodeInput'].to(device)
            decoderInput = batch['decodeInput'].to(device)
            encoderMask = batch['encoderMask'].to(device)
            decoderMask = batch['decoderMask'].to(device)
            
            encoderOutput = model.encode(encoderInput, encoderMask)
            decoderOutput = model.decode(encoderOutput, encoderMask, decoderInput, decoderMask)
            projectionOutput = model.project(decoderOutput)
            
            label = batch['label'].to(device)
            
            loss = loss_function(projectionOutput.view(-1, targetTokenizer.get_vocab_size()), label.view(-1))
            
            batchIterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
if __name__ == '__main__':
    trainModel()