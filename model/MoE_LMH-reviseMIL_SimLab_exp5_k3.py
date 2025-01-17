import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from openprompt.data_utils import InputExample
import csv
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import random_split
from torch import Tensor
from openprompt.prompts import ManualTemplate
from clone import clone_module
import gc
from transformers import BertConfig
import copy

#graphic card classes and label words
graphic_classes = [
    0,
    1,
    2,
    3,
    4,
    5
]
graphic_label_words={
    0:['A'],
    1:['B'],
    2:['C'],
    3:['D'],
    4:['E'],
    5:['F'],
}
cpu_classes = [
    0,
    1,
    2,
    3,
    4
]
cpu_label_words={
    0:['A'],
    1:['B'],
    2:['C'],
    3:['D'],
    4:['E'],
}
hard_classes = [
    0,
    1,
    2,
    3,
    4,
    5
]
hard_label_words={
    0:['A'],
    1:['B'],
    2:['C'],
    3:['D'],
    4:['E'],
    5:['F'],
}
ram_classes = [
    0,
    1,
    2,
    3,
    4
]
ram_label_words={
    0:['A'],
    1:['B'],
    2:['C'],
    3:['D'],
    4:['E'],
}
scre_classes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
]
scre_label_words={
    0:['A'],
    1:['B'],
    2:['C'],
    3:['D'],
    4:['E'],
    5:['F'],
    6:['G'],
    7:['H'],
}

def copy_batch(batch,device):
    n_batch={}
    for key in batch.keys():
        ni=torch.tensor(batch[key],device=device)
        n_batch[key]=ni
    return n_batch
        

def read_data_csv(file,ratio):
    record=[]
    with open(file,newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            record.append(item[1:])
    record=record[1:]
    for ind,sample in enumerate(record):
        sample.insert(0,ind)
        sample[2]=int(sample[2])#cpu
        sample[3]=int(sample[3])#graphic
        sample[4]=int(sample[4])#hardisk
        sample[5]=int(sample[5])#ram
        sample[6]=int(sample[6])#screen
    train_set, valid_set=random_split(record,
                 #[0.7,0.3],
                 ratio,
                 generator=torch.Generator().manual_seed(42))
    dataset={}
    train_dataset=[]
    valid_dataset=[]
    for item in train_set:
        train_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2:]))
    for item in valid_set:
        valid_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2:]))
    dataset['train']=train_dataset
    dataset['valid']=valid_dataset
    return dataset


class multiMask_MixTemplateModel(nn.Module):
    def __init__(self,
                plm:PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                WrapperClass,
                dataset,
                needdata,
                num_expert,
                #classes,
                epoch,
                template_text,
                #shareTemplate,
                #label_words,
                device,
                cpu_classes,
                cpu_label_words,
                graphic_classes,
                grapihc_label_words,
                hardisk_classes,
                hardisk_label_words,
                ram_classes,
                ram_label_words,
                screen_classes,
                screen_label_words,
                ):
        
        super().__init__()
        self.device=device
        
        #self.promptTemplate = shareTemplate
        self.promptTemplate = MixedTemplate(
            model=plm,
            text = template_text,
            tokenizer = tokenizer,
        )

        # 5 verbalizer correpond to 5 attributes(cpu, graphic card, hard disk, ram, screen)
        self.cpu_promptVerbalizer = ManualVerbalizer(
            classes = cpu_classes,
            label_words = cpu_label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )
        self.cpu_promptVerbalizer.to(device)
        self.graphic_promptVerbalizer = ManualVerbalizer(
            classes = graphic_classes,
            label_words = graphic_label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )
        self.graphic_promptVerbalizer.to(device)
        self.hardisk_promptVerbalizer = ManualVerbalizer(
            classes = hardisk_classes,
            label_words = hardisk_label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )
        self.hardisk_promptVerbalizer.to(device)
        self.ram_promptVerbalizer = ManualVerbalizer(
            classes = ram_classes,
            label_words = ram_label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )
        self.ram_promptVerbalizer.to(device)
        self.screen_promptVerbalizer = ManualVerbalizer(
            classes = screen_classes,
            label_words = screen_label_words,
            tokenizer = tokenizer,
            multi_token_handler="mean",
        )
        self.screen_promptVerbalizer.to(device)

        # Model backbone
        self.promptModel = PromptForClassification(
            template = self.promptTemplate,
            plm = plm,
            #verbalizer = self.promptVerbalizer,
            verbalizer = None,#rewrite model foward function and use the verbalizer outside
        )
        self.promptModel.to(device)

        #train_set, valid_set=random_split(dataset,
        #                                  [0.7,0.3],
        #                                  generator=torch.Generator().manual_seed(42))
        train_set=dataset['train']
        valid_set=dataset['valid']
        
        finetune_set=needdata['train']
        test_set=needdata['valid']

        self.train_data_loader = PromptDataLoader(
            dataset = train_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            shuffle=True,
            #max_seq_length=800,
        )
        self.valid_data_loader = PromptDataLoader(
            dataset = valid_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            #max_seq_length=800,
        )
        
        self.finetune_data_loader = PromptDataLoader(
            dataset = finetune_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            shuffle=True,
            #max_seq_length=800,
        )
        self.test_data_loader = PromptDataLoader(
            dataset = test_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            #max_seq_length=800,
        )
        
        self.k = 3
        self.num_expert = num_expert
        self.MoELMH = []#The list containing all expert language modeling heads
        for i in range(num_expert):
            #clone_LMH = type(plm.cls)(BertConfig)
            #clone_LMH = nn.Sequential(
            #    nn.Linear(in_features=768, out_features=768, bias=True),
            #    nn.GELU(),
            #    nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True),
            #    nn.Linear(in_features=768, out_features=28996, bias=True)
            #)
            #clone_LMH.load_state_dict(plm.cls.state_dict())
            
            #model_copy = type(mymodel)() # get a new instance
            #model_copy.load_state_dict(mymodel.state_dict())
            
            #self.MoELMH.append(clone_module(plm.cls))
            self.MoELMH.append(copy.deepcopy(plm.cls))
            self.MoELMH[-1].load_state_dict(plm.cls.state_dict())
            #self.MoELMH.append(clone_LMH)
            self.MoELMH[-1].to(device)
            
            #for n,p in self.MoELMH[-1].named_parameters():
            #    print(p.data_ptr())
            #print('--------------------------')
        
        #for i in range(num_expert):
        #    for n, p in self.MoELMH[i].named_parameters():
        #        print('hjh check is leaf: ',n,', ',p.is_leaf)
        #for n, p in self.promptModel.plm.named_parameters():
        #    print('hjh check is leaf: ',n,', ',p.is_leaf)
        
        
        self.w_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(768, num_expert, bias=False)
        )
        self.w_gate.to(device)

        self.cross_entropy  = nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        for i in range(num_expert):
            optimizer_grouped_parameters1.append(
                {'params': [p for n, p in self.MoELMH[i].named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
            )
            optimizer_grouped_parameters1.append(
                {'params': [p for n, p in self.MoELMH[i].named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            )
        
        # Using different optimizer for prompt parameters and model parameters
        optimizer_grouped_parameters2 = [
            {'params': [p for n,p in self.promptModel.template.named_parameters() if "raw_embedding" not in n]}
        ]
        optimizer_grouped_parameters3 = [
            {'params': [p for n,p in self.w_gate.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n,p in self.w_gate.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-5)
        self.optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-5)
        self.optimizer3 = AdamW(optimizer_grouped_parameters3, lr=1e-4)
        
        self.epoch=epoch

    def forward(self,batch):
        """outputs=self.promptModel(batch)"""
        #rewrite forward function
        
        outputs = self.promptModel.prompt_model(batch)
        
        #outputs = self.verbalizer.gather_outputs(outputs)
        """replace the following instruction with a MoE-supported operation"""
        #outputs=outputs.logits
        outputs, miloss = self.MoELM(outputs, batch)
        
        """if isinstance(outputs, tuple):
            outputs_at_mask = [self.promptModel.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.promptModel.extract_at_mask(outputs, batch)"""
        
        #use 5 verbalizers to replace original one
        #label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        outputs_at_mask=torch.transpose(outputs,0,1)
        #cpu: 0/1/2/3
        #graphic: 0/4/5/6
        #hardisk: 1/4/7/8
        #ram: 2/5/7/9
        #screen: 3/6/8/9
        #cpu_outputs_at_mask=outputs_at_mask[0+5]+outputs_at_mask[1+5]+outputs_at_mask[2+5]+outputs_at_mask[3+5]+outputs_at_mask[10-10]
        cpu_outputs_at_mask=outputs_at_mask[10-10]
        cpu_label_words_logits = self.cpu_promptVerbalizer.process_outputs(cpu_outputs_at_mask, batch=batch)
        
        #graphic_outputs_at_mask=outputs_at_mask[0+5]+outputs_at_mask[4+5]+outputs_at_mask[5+5]+outputs_at_mask[6+5]+outputs_at_mask[11-10]
        graphic_outputs_at_mask=outputs_at_mask[11-10]
        graphic_label_words_logits = self.graphic_promptVerbalizer.process_outputs(graphic_outputs_at_mask, batch=batch)
        
        #hardisk_outputs_at_mask=outputs_at_mask[1+5]+outputs_at_mask[4+5]+outputs_at_mask[7+5]+outputs_at_mask[8+5]+outputs_at_mask[12-10]
        hardisk_outputs_at_mask=outputs_at_mask[12-10]
        hardisk_label_words_logits = self.hardisk_promptVerbalizer.process_outputs(hardisk_outputs_at_mask, batch=batch)
        
        #ram_outputs_at_mask=outputs_at_mask[2+5]+outputs_at_mask[5+5]+outputs_at_mask[7+5]+outputs_at_mask[9+5]+outputs_at_mask[13-10]
        ram_outputs_at_mask=outputs_at_mask[13-10]
        ram_label_words_logits = self.ram_promptVerbalizer.process_outputs(ram_outputs_at_mask, batch=batch)
        
        #screen_outputs_at_mask=outputs_at_mask[3+5]+outputs_at_mask[6+5]+outputs_at_mask[8+5]+outputs_at_mask[9+5]+outputs_at_mask[14-10]
        screen_outputs_at_mask=outputs_at_mask[14-10]
        screen_label_words_logits = self.screen_promptVerbalizer.process_outputs(screen_outputs_at_mask, batch=batch)
        
        return cpu_label_words_logits, graphic_label_words_logits, hardisk_label_words_logits,\
                ram_label_words_logits, screen_label_words_logits, miloss

    def MoELM(self, raw_outputs, batch):
        #bsz, mask_num, emb_size = raw_outputs.size()
        #raw_outputs_repre shape: [batch_size, mask_num, hidden_layer_dimension:768]
        raw_outputs_repre = self.promptModel.extract_at_mask(raw_outputs.hidden_states[12],batch)
        bsz, mask_num, emb_size = raw_outputs_repre.size()
        """raw_outputs_repre = raw_outputs_repre.reshape(-1, emb_size)"""
        
        top_k_gates, top_k_indices, miloss = self.top_k_gating(raw_outputs_repre)
        shape=[self.num_expert, bsz, mask_num, 28996]
        raw_outputs = torch.zeros(shape, device=self.device)
        for i in range(self.num_expert):
            raw_outputs[i] = self.MoELMH[i](raw_outputs_repre)
            #print(raw_outputs[i])
        #for i in range(self.num_expert):
        #    for j in range(i+1,self.num_expert):
        #        if(not torch.equal(raw_outputs[i],raw_outputs[j])):
        #            print('hjh found err!')
        #            print(i,j)
                    #print(raw_outputs[i])
                    #print(raw_outputs[j])
        #print('-------------------------------')
        shape=[bsz, mask_num, 28996]
        MoE_outputs = torch.zeros(shape, device=self.device)
        for i in range(bsz):
            for j in range(mask_num):
                for expert in range(len(top_k_indices[i][j])):
                    expert_id=top_k_indices[i][j][expert]
                    MoE_outputs[i][j] = MoE_outputs[i][j] + \
                                top_k_gates[i][j][expert] * \
                                raw_outputs[expert_id][i][j]
        
        return MoE_outputs, miloss
    
    def top_k_gating(self, x):
        logits = self.w_gate(x)

        probs = torch.softmax(logits, dim=2)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=2)
        
        #print(top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6))
        #print('---------------------------------------')
        #print(top_k_gates / (top_k_gates.sum(dim=2, keepdim=True) + 1e-6))
        #print('---------------------------------------')
        
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=2, keepdim=True) + 1e-6)
        #top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True))

        return top_k_gates, top_k_indices, self.compute_miloss(logits, probs, None)
    
    def compute_miloss(self, logits, prob, mask=None):
        bsz, mask_num, num_expert = logits.size()
        logits = logits.reshape(-1,num_expert)
        prob = prob.reshape(-1,num_expert)
        
        log_prob = torch.log_softmax(logits, dim=-1)
        if mask is None:
            p_x = 1 / logits.shape[0]
        else:
            p_x = mask / (mask.sum() + 1e-12)
        p_e = (p_x * prob).sum(0)
        H_e = (p_e * p_e.log()).sum()
        neg_H_e_given_x = (p_x * prob * log_prob).sum()
        miloss = -neg_H_e_given_x + H_e

        return miloss

    def forward_MoE(self, input: Tensor, expert_size: Tensor, num_expert: int, MoELMH):
        output_buf: Tensor = torch.empty((input.size(0), 28996),
                                        device=input.device, dtype=input.dtype)
        #num_linears = num_expert

        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)
        output_buf_list = list(output_buf_list)

        for i in range(num_expert):
        #    torch.mm(input_list[i], weight[i], out=output_buf_list[i])
            output_buf_list[i]=MoELMH[i](input_list[i])

        #output = output_buf
        output = torch.cat(output_buf_list)
        return output
    
    def train(self):
        self.promptModel.train()
        self.w_gate.train()

    def eval(self):
        self.promptModel.eval()
        self.w_gate.eval()
    
    def set_epoch(self,epoch):
        self.epoch=epoch






if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    
    num_expert = 5
    
    #assert(1==2)
    
    #dataset=read_data_csv("newdata/clean_new_processed_review_all_map.csv",[16274,6975])
    #dataset=read_data_csv("newdata/new_need_all_map.csv",[1149,1148])
    dataset=read_data_csv("newdata/new_review_all_map.csv",[17018,7293])
    need_dataset=read_data_csv("newdata/new_need_all_map.csv",[1149,1148])
    
    log_file_name='./result/MoELMH_exp5_k3_reviseMIL_SimLab/run2/log_run2.txt'
    
    epoch=20
    template='{"soft": "unused unused unused unused"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"placeholder":"text_a"} {"soft": "unused unused"}'
    
    
    model=multiMask_MixTemplateModel(plm,
                                   tokenizer,
                                   WrapperClass,
                                   dataset,
                                   need_dataset,
                                   num_expert,
                                   epoch,
                                   template,
                                   device,
                                   cpu_classes,
                                    cpu_label_words,
                                    graphic_classes,
                                    graphic_label_words,
                                    hard_classes,
                                    hard_label_words,
                                    ram_classes,
                                    ram_label_words,
                                    scre_classes,
                                    scre_label_words)
    

    
    #-----------------------Train-------------------------
    #model.train()
    for i in range(model.epoch):
        count=0
        loss_rec=0
        model.train()
        for batch in model.train_data_loader:
            batch.to(device)
            
            labels=batch['label']
            label_trans=torch.transpose(batch['label'],0,1)
            cpu_labels=label_trans[0]
            graphic_labels=label_trans[1]
            hard_labels=label_trans[2]
            ram_labels=label_trans[3]
            scre_labels=label_trans[4]
            
            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits, miloss = model(batch)
            
            cpu_loss=model.cross_entropy(cpu_logits,cpu_labels)
            graphic_loss=model.cross_entropy(graphic_logits,graphic_labels)
            hard_loss=model.cross_entropy(hard_logits,hard_labels)
            ram_loss=model.cross_entropy(ram_logits,ram_labels)
            scre_loss=model.cross_entropy(scre_logits,scre_labels)
            
            shared_loss=cpu_loss+graphic_loss+hard_loss+ram_loss+scre_loss+miloss
            
            shared_loss.backward()
            
            model.optimizer1.step()
            model.optimizer1.zero_grad()
            
            model.optimizer2.step()
            model.optimizer2.zero_grad()
            
            model.optimizer3.step()
            model.optimizer3.zero_grad()
            
            count+=1
            loss_rec+=shared_loss
            #break
            
            
        gc.collect()
        torch.cuda.empty_cache()
        print('NO.',i,' epoch avg loss: ',loss_rec/count)
        
        f = open(log_file_name, "a")
        f.write('NO.'+str(i)+' epoch avg loss: '+str(loss_rec/count)+'\n')
        f.close()
    
        #save checkpoint
        if(i==4 or i==9 or i==14 or i==19 or i==24 or i==29):
            with torch.no_grad():
                model.eval()
                cpu_preds=[]
                cpu_labels=[]
                cpu_all_pred=[]
                graphic_preds=[]
                graphic_labels=[]
                graphic_all_pred=[]
                hard_preds=[]
                hard_labels=[]
                hard_all_pred=[]
                ram_preds=[]
                ram_labels=[]
                ram_all_pred=[]
                scre_preds=[]
                scre_labels=[]
                scre_all_pred=[]
                for step, inputs in enumerate(model.test_data_loader):
                    inputs.to(device)
                    cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits, _=model(inputs)

                    label_trans=torch.transpose(inputs['label'],0,1)
                    cpu_label=label_trans[0]
                    graphic_label=label_trans[1]
                    hard_label=label_trans[2]
                    ram_label=label_trans[3]
                    scre_label=label_trans[4]

                    cpu_labels.extend(cpu_label.cpu().tolist())
                    cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
                    cpu_all_pred.extend(cpu_logits.cpu().tolist())
                    graphic_labels.extend(graphic_label.cpu().tolist())
                    graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
                    graphic_all_pred.extend(graphic_logits.cpu().tolist())
                    hard_labels.extend(hard_label.cpu().tolist())
                    hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
                    hard_all_pred.extend(hard_logits.cpu().tolist())
                    ram_labels.extend(ram_label.cpu().tolist())
                    ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
                    ram_all_pred.extend(ram_logits.cpu().tolist())
                    scre_labels.extend(scre_label.cpu().tolist())
                    scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
                    scre_all_pred.extend(scre_logits.cpu().tolist())
                    #break

                cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
                graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
                hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
                ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
                scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)
            
                f = open(log_file_name, "a")
                f.write(str(i)+" epoch test cpu accuracy is : "+str(cpu_acc)+'\n')
                f.write(str(i)+" epoch test graphic card accuracy is : "+str(graphic_acc)+'\n')
                f.write(str(i)+" epoch test hard disk accuracy is : "+str(hard_acc)+'\n')
                f.write(str(i)+" epoch test ram accuracy is : "+str(ram_acc)+'\n')
                f.write(str(i)+" epoch test scre accuracy is : "+str(scre_acc)+'\n')
                f.close()
                
                print(i," epoch test cpu accuracy is : ",cpu_acc)
                print(i," epoch test graphic card accuracy is : ",graphic_acc)
                print(i," epoch test hard disk accuracy is : ",hard_acc)
                print(i," epoch test ram accuracy is : ",ram_acc)
                print(i," epoch test scre accuracy is : ",scre_acc)
            
            PATH = "./../autodl-tmp/checkpoint/MoELMH_exp5_k3_reviseMIL_SimLab/run2/MoELMH_exp5_k3_reviseMIL_SimLab_CEL_Epoch_"+str(i+1)+".pt"
            torch.save({
                    'epoch': i+1,
                    'plm_state_dict': model.promptModel.plm.state_dict(),
                    'template_state_dict': model.promptModel.template.state_dict(),
                    'optimizer1_state_dict': model.optimizer1.state_dict(),
                    'optimizer2_state_dict': model.optimizer2.state_dict(),
                    'optimizer3_state_dict': model.optimizer3.state_dict(),
                    'loss': shared_loss,
                    'w_gate': model.w_gate.state_dict(),
                    'expert0': model.MoELMH[0].state_dict(),
                    'expert1': model.MoELMH[1].state_dict(),
                    'expert2': model.MoELMH[2].state_dict(),
                    'expert3': model.MoELMH[3].state_dict(),
                    'expert4': model.MoELMH[4].state_dict(),
                    }, PATH)
    
    #assert(1==2,'test success!!!')
    print('test success!!!')
    #-----------------------Validate-------------------------
    model.eval()
    cpu_preds=[]
    cpu_labels=[]
    graphic_preds=[]
    graphic_labels=[]
    hard_preds=[]
    hard_labels=[]
    ram_preds=[]
    ram_labels=[]
    scre_preds=[]
    scre_labels=[]
    with torch.no_grad():
        model.eval()
        #for step, inputs in enumerate(graphic_model.valid_data_loader):
        for step, inputs in enumerate(model.valid_data_loader):
            inputs.to(device)

            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits, _ = model(inputs)

            label_trans=torch.transpose(inputs['label'],0,1)
            cpu_label=label_trans[0]
            graphic_label=label_trans[1]
            hard_label=label_trans[2]
            ram_label=label_trans[3]
            scre_label=label_trans[4]

            cpu_labels.extend(cpu_label.cpu().tolist())
            cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
            graphic_labels.extend(graphic_label.cpu().tolist())
            graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
            hard_labels.extend(hard_label.cpu().tolist())
            hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
            ram_labels.extend(ram_label.cpu().tolist())
            ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
            scre_labels.extend(scre_label.cpu().tolist())
            scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
            #break
            
        
    cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
    graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
    hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
    ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
    scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)
    
    f = open(log_file_name, "a")
    f.write(" validate cpu accuracy is : "+str(cpu_acc)+'\n')
    f.write(" validate graphic card accuracy is : "+str(graphic_acc)+'\n')
    f.write(" validate hard disk accuracy is : "+str(hard_acc)+'\n')
    f.write(" validate ram accuracy is : "+str(ram_acc)+'\n')
    f.write(" validate scre accuracy is : "+str(scre_acc)+'\n')
    f.close()
    
    print("cpu accuracy is : ",cpu_acc)
    print("graphic card accuracy is : ",graphic_acc)
    print("hard disk accuracy is : ",hard_acc)
    print("ram accuracy is : ",ram_acc)
    print("scre accuracy is : ",scre_acc)

    
    #-----------------------Fine tune-------------------------
    #model.train()
    for i in range(20):
        count=0
        loss_rec=0
        model.train()
        for batch in model.finetune_data_loader:
            batch.to(device)
            
            labels=batch['label']
            label_trans=torch.transpose(batch['label'],0,1)
            cpu_labels=label_trans[0]
            graphic_labels=label_trans[1]
            hard_labels=label_trans[2]
            ram_labels=label_trans[3]
            scre_labels=label_trans[4]
            
            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits, miloss = model(batch)
            
            cpu_loss=model.cross_entropy(cpu_logits,cpu_labels)
            graphic_loss=model.cross_entropy(graphic_logits,graphic_labels)
            hard_loss=model.cross_entropy(hard_logits,hard_labels)
            ram_loss=model.cross_entropy(ram_logits,ram_labels)
            scre_loss=model.cross_entropy(scre_logits,scre_labels)
            
            shared_loss=cpu_loss+graphic_loss+hard_loss+ram_loss+scre_loss+miloss
            
            shared_loss.backward()
            
            model.optimizer1.step()
            model.optimizer1.zero_grad()
            
            model.optimizer2.step()
            model.optimizer2.zero_grad()
            
            model.optimizer3.step()
            model.optimizer3.zero_grad()
            
            count+=1
            loss_rec+=shared_loss
            #break
            
            
        gc.collect()
        torch.cuda.empty_cache()
        print('NO.',i,' epoch avg loss: ',loss_rec/count)
        
        f = open(log_file_name, "a")
        f.write('NO.'+str(i+1)+' epoch avg loss: '+str(loss_rec/count)+'\n')
        f.close()
        
        #-----------------------test-------------------------
        if(i==4 or i==9 or i==14 or i==19):
            with torch.no_grad():
                model.eval()
                cpu_preds=[]
                cpu_labels=[]
                cpu_all_pred=[]
                graphic_preds=[]
                graphic_labels=[]
                graphic_all_pred=[]
                hard_preds=[]
                hard_labels=[]
                hard_all_pred=[]
                ram_preds=[]
                ram_labels=[]
                ram_all_pred=[]
                scre_preds=[]
                scre_labels=[]
                scre_all_pred=[]
                for step, inputs in enumerate(model.test_data_loader):
                    inputs.to(device)
                    cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits, _=model(inputs)

                    label_trans=torch.transpose(inputs['label'],0,1)
                    cpu_label=label_trans[0]
                    graphic_label=label_trans[1]
                    hard_label=label_trans[2]
                    ram_label=label_trans[3]
                    scre_label=label_trans[4]

                    cpu_labels.extend(cpu_label.cpu().tolist())
                    cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
                    cpu_all_pred.extend(cpu_logits.cpu().tolist())
                    graphic_labels.extend(graphic_label.cpu().tolist())
                    graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
                    graphic_all_pred.extend(graphic_logits.cpu().tolist())
                    hard_labels.extend(hard_label.cpu().tolist())
                    hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
                    hard_all_pred.extend(hard_logits.cpu().tolist())
                    ram_labels.extend(ram_label.cpu().tolist())
                    ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
                    ram_all_pred.extend(ram_logits.cpu().tolist())
                    scre_labels.extend(scre_label.cpu().tolist())
                    scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
                    scre_all_pred.extend(scre_logits.cpu().tolist())
                    #break
                    
                cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
                graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
                hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
                ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
                scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)

            save_path="./result/MoELMH_exp5_k3_reviseMIL_SimLab/run2/MoELMH_exp5_k3_CEL_all_epoch_"+str(i+1)+"_test_res.csv"
            n=len(cpu_labels)
            record=[]
            for j in range(0,n):
                tmp={"index":j, "cpu_label":cpu_labels[j], "cpu_prediction":cpu_preds[j], "cpu_all_pred": cpu_all_pred[j],\
                                "graphic_label":graphic_labels[j], "graphic_prediction":graphic_preds[j], "graphic_all_pred": graphic_all_pred[j],\
                                "hard_label":hard_labels[j], "hard_prediction":hard_preds[j], "hard_all_pred": hard_all_pred[j],\
                                "ram_label":ram_labels[j], "ram_prediction":ram_preds[j], "ram_all_pred": ram_all_pred[j],\
                                "screen_label":scre_labels[j], "screen_prediction":scre_preds[j], "screen_all_pred": scre_all_pred[j]}
                record.append(tmp)

            with open(save_path, 'w', newline='') as csvfile:
                fieldnames = ['index', 'cpu_label','cpu_prediction','cpu_all_pred',\
                             'graphic_label','graphic_prediction','graphic_all_pred',\
                             'hard_label','hard_prediction','hard_all_pred',\
                             'ram_label','ram_prediction','ram_all_pred',\
                             'screen_label','screen_prediction','screen_all_pred']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(record)
            print(save_path)
            print(i," epoch test cpu accuracy is : ",cpu_acc)
            print(i," epoch test graphic card accuracy is : ",graphic_acc)
            print(i," epoch test hard disk accuracy is : ",hard_acc)
            print(i," epoch test ram accuracy is : ",ram_acc)
            print(i," epoch test scre accuracy is : ",scre_acc)
            
            f = open(log_file_name, "a")
            f.write(str(i+1)+" epoch test cpu accuracy is : "+str(cpu_acc)+'\n')
            f.write(str(i+1)+" epoch test graphic card accuracy is : "+str(graphic_acc)+'\n')
            f.write(str(i+1)+" epoch test hard disk accuracy is : "+str(hard_acc)+'\n')
            f.write(str(i+1)+" epoch test ram accuracy is : "+str(ram_acc)+'\n')
            f.write(str(i+1)+" epoch test scre accuracy is : "+str(scre_acc)+'\n')
            f.close()