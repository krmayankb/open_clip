import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_xla.core.xla_model as xm

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template


def zero_shot_classifier(model, classnames, templates, dim, args):
    tokenizer = get_tokenizer(args.model)
    xm.mark_step()
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            #if args.distributed and not args.horovod:
            #    class_embeddings = model.module.encode_text(texts)
            #else:
            xm.mark_step()
            class_embeddings = model.encode_text(texts)
            xm.mark_step()
            if args.force_mrl_loss:
                class_embeddings = class_embeddings[:,:dim] # consider only dim for creating zeroshot classifier 
            xm.mark_step()
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            xm.mark_step()
            class_embedding /= class_embedding.norm()
            xm.mark_step()
            zeroshot_weights.append(class_embedding)
        xm.mark_step()
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    xm.mark_step()
    return zeroshot_weights


import lightning as L 
def accuracy(output, target, topk=(1,)):
    xm.mark_step()
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    xm.mark_step()
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, dim, args):
    xm.mark_step()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        xm.mark_step()
        top1, top5, n = 0., 0., 0.
        #print("JUST BEFORE FOR LOOP")
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
        #for _ in tqdm(range(5), unit_scale = args.batch_size):
           # print("INSIDE RUN, TQDM DATALOADER")
            #images, target = next(iter(dataloader))
            images = images.to(args.device)
            xm.mark_step()
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            xm.mark_step()
            target = target.to(args.device)
            #print("JUST BEFORE AUTOCAST")
            with autocast():
                #print("JUST INSIDE AUTOCAST")
                # predict
                #if args.distributed and not args.horovod:
                #    image_features = model.module.encode_image(images)
                #else:
                image_features = model.encode_image(images)
                #print("DIM OF IMAGE FEATURES: ", image_features.shape)
                if args.force_mrl_loss:
                    image_features = image_features[:,:dim]
                #print("DIM OF IMAGE AFTER MRL SLICE: ", image_features.shape)
                image_features = F.normalize(image_features, dim=-1)
                #print("DIM OF IMAGE AFTER MRL SLICE: ", image_features.shape)
                logits = 100. * image_features @ classifier
                #print("AFTER LOGIT")
            # measure accuracy
            xm.mark_step()
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            xm.mark_step()
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            del image_features, images, target

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    #print("INSIDE ZEROSHOT EVALUATION")
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    # if MRL 
    #print("INSIDE ZEROSHOT, BEFORE FORCE_MRL")
    if args.force_mrl_loss:
        results = {}
        for dim in args.mrl_dim_to_consider: 
            logging.info(f'Building zero-shot classifier dim-{dim}')
            classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, dim, args)

            logging.info('Using classifier')
            if 'imagenet-val' in data:
                #top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, dim, args)
                top1, top5 = run(model, classifier, args.fabric_val_dataloader, dim, args)
                top1_name = f'imagenet-zeroshot-val-d{dim}-top1'
                top5_name = f'imagenet-zeroshot-val-d{dim}-top5'
                #print(top1_name, top1.shape, top5.shape)
                
                results[top1_name] = top1
                results[top5_name] = top5
            if 'imagenet-v2' in data:
                top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, dim, args)
                top1_name = f'imagenet-zeroshot-val-d{dim}-top1'
                top5_name = f'imagenet-zeroshot-val-d{dim}-top5'
                results[top1_name] = top1
                results[top5_name] = top5
    
    # for other losses than MRL
    else: 
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, 0, args)

        logging.info('Using classifier')
        results = {}
        if 'imagenet-val' in data:
            top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, 0, args)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
        if 'imagenet-v2' in data:
            top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, 0, args)
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

    # if not MRL 
    logging.info('Finished zero-shot imagenet.')
    return results
