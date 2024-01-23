import argparse
from transformers import BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
from transformers import BertConfig, BertModel
import torch.nn.utils.prune as prune
import numpy as np  
import torch  
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--weight', default='pre', type=str, help='file_dir')
parser.add_argument('--model', default='glue', type=str, help='file_dir')
parser.add_argument('--rate', default=0.2, type=float, help='rate')
args = parser.parse_args()



def pruning_model(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_weight_rate(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))


    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list


config = BertConfig.from_pretrained(
    'bert-base-uncased'
)


if args.model == 'glue':

    if args.weight == 'rand':
        print('random')
        model = BertForSequenceClassification(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = AutoModelForSequenceClassification.from_pretrained("gchhablani/bert-base-cased-finetuned-mrpc")
        # model = BertForSequenceClassification.from_pretrained(
        #     'bert-base-uncased',
        #     from_tf=bool(".ckpt" in 'bert-base-uncased'),
        #     config=config
        # )
        output = 'pruned_model/'
    # print(len(model.state_dict().keys()))
    pruning_model(model, args.rate)
    # print(model.state_dict().keys())
    zero = see_weight_rate(model)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
        else:
            weight_dict[key] = model_dict[key]

    torch.save(mask_dict, output+'mask.pt')
    torch.save(weight_dict, output+'weight.pt')

    # mask_dict = {}
    # weight_dict = {}
    # keys_to_remove = []
    # keys_to_rename = []
    # model_dict = model.state_dict()
    # for key in model_dict.keys():
    #     if 'mask' in key:
    #         # print("Mask: ", model_dict[key].shape)
    #         mask = model_dict[key]
    #         new_key = key.replace("_mask", "")
    #         orig_key = new_key + '_orig'
    
    #         # print("Weight: ",model_dict[key].shape)
    #         model_dict[orig_key] *= mask
    #         # print(model_dict[orig_key])
    #         keys_to_remove.append(key)
    #         keys_to_rename.append(orig_key)
    
    # for key in keys_to_remove:
    #       del model_dict[key]
    # for key in keys_to_rename:
    #     new_key = key.replace("_orig", "")
    #     model_dict[new_key] = model_dict.pop(key)



    # torch.save(model_dict, output+str(args.rate)+'pruned_model.pth')


elif args.model == 'squad':

    if args.weight == 'rand':
        print('random')
        model = BertForQuestionAnswering(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = BertForQuestionAnswering.from_pretrained(
            'bert-base-uncased',
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
        output = 'pretrain_prun/'

    pruning_model(model, args.rate)
    zero = see_weight_rate(model)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
        else:
            weight_dict[key] = model_dict[key]

    # torch.save(mask_dict, output+'mask.pt')
    # torch.save(weight_dict, output+'weight.pt')
    torch.save(model.state_dict(), output+'pruned_model.pth')

elif args.model == 'pretrain':

    if args.weight == 'rand':
        print('random')
        model = BertForMaskedLM(config=config)
        output = 'random_prun/'

    elif args.weight == 'pre':
        model = BertForMaskedLM.from_pretrained(
            'bert-base-uncased',
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
        output = 'pretrain_prun/'

    pruning_model(model, args.rate)
    zero = see_weight_rate(model)
    print('zero rate', zero)

    mask_dict = {}
    weight_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
        else:
            weight_dict[key] = model_dict[key]

    # torch.save(mask_dict, output+'mask.pt')
    # torch.save(weight_dict, output+'weight.pt')
    torch.save(model.state_dict(), output)

