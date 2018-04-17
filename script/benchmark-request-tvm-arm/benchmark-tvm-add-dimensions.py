#! /usr/bin/python
import ck.kernel as ck
import os

def do(i):

    # List performance entries
    r=ck.access({'action':'search',
                 'module_uoa':'experiment',
                 'data_uoa':'ck-request-asplos18-nnvm-tvm-arm-performance*'
#                 'repo_uoa':'ck-request-asplos18-results'
                })
    if r['return']>0: return r
    lst=r['lst']

    for q in lst:
        duid=q['data_uid']
        duoa=q['data_uoa']
        ruid=q['repo_uid']
        path=q['path']

        ck.out(duoa)

        # Search matching accuracy entry
        r=ck.access({'action':'load',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid})
        if r['return']>0: return r

        dd=r['dict']
        ruid=r['repo_uid']
        apath=r['path']             

        # Updating meta if needed
        dd['meta']['scenario_module_uoa']='a555738be4b65860' # module:request.asplos18

        dd['meta']['dataset_species']='ImageNet' # dataset species (free format)

        dd['meta']['platform_species']='embedded' # embedded vs server (maybe other classifications such as edge)

        dd['meta']['platform_peak_power']=6.05 #Watts http://opensource.rock-chips.com/images/6/60/Rockchip_RK3399_Datasheet_V1.6-20170301.pdf last page
        dd['meta']['platform_price']=149 # $, http://shop.t-firefly.com/goods.php?id=45
        dd['meta']['platform_price_date']='20180416' # date

        dd['meta']['artifact']='08da9685582866a0' # artifact description

        dd['meta']['processed']='yes'

        xpr=dd['meta'].get('model_precision','')
        mult=1.0
        if xpr=='fp16': mult=0.5

        # Unified full name for some deps
        ds=dd['meta']['deps_summary']

        x=ds['mxnet-model']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        full_model_name=r['full_name']

        dd['meta']['model_design_name']=full_model_name

        # for simplicity add manually (can later automate it as in other artifacts, but just didn't have time here)
        if 'mobilenet' in full_model_name:
           dd['meta']['model_species']='07d4e7aa3750ddc6' # model.species:resnet18
           dd['meta']['dataset_size']=50000 # number of images ...
           accuracy_top1=0.66694
           accuracy_top5=0.87734
        elif 'resnet' in full_model_name:
           dd['meta']['model_species']='d41bbf1e489ab5e0' # model.species:resnet18
           dd['meta']['dataset_size']=25000 # number of images ...
           accuracy_top1=0.61318
           accuracy_top5=0.83702
        elif 'vgg16' in full_model_name:
           dd['meta']['model_species']='a3fcac86d42bdbc4' # model.species:resnet18
           dd['meta']['dataset_size']=5000 # number of images ...
           accuracy_top1=0.63120
           accuracy_top5=0.84951
        else:
           return {'return':1, 'error':'unknown model ('+y+')'}

        x=ds['lib-mxnet']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        dd['meta']['library_name']=r['full_name']

        x=x['deps']['compiler']
        r=ck.access({'action':'make_deps_full_name','module_uoa':'request.asplos18','deps':x})
        if r['return']>0: return r
        dd['meta']['compiler_name']=r['full_name']

        # Updating entry
        r=ck.access({'action':'update',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid,
                     'dict':dd,
                     'substitute':'yes',
                     'ignore_update':'yes',
                     'sort_keys':'yes'
                    })
        if r['return']>0: return r

        # Checking points to aggregate
        os.chdir(path)
        dperf=os.listdir(path)
        for f in dperf:
            if f.endswith('.cache.json'):
               os.system('git rm -f '+f)

            elif f.endswith('.flat.json'):
               ck.out(' * '+f)

               # Load performance file 
               p1=os.path.join(path, f)

               r=ck.load_json_file({'json_file':p1})
               if r['return']>0: return r
               d1=r['dict']

               # Prune some old value
               d={}
               for k in d1:
                   if not k.startswith('##characteristics#run#accuracy_top1') and \
                      not k.startswith('##characteristics#run#accuracy_top5') and \
                      not k.startswith('##characteristics#run#inference_throughput') and \
                      not k.startswith('##characteristics#run#inference_latency'):
                      d[k]=d1[k]

               # for simplicity add manually (can later automate it as in other artifacts, but just didn't have time here)
               if 'mobilenet' in full_model_name:
                  model_size=17024109*mult
               elif 'resnet' in full_model_name:
                  model_size=46803089*mult
               elif 'vgg16' in full_model_name:
                  model_size=553432060*mult
               else:
                  return {'return':1, 'error':'unknown model ('+y+')'}

               d['##features#model_size#min']=model_size # Bytes

               d['##features#gpu_freq#min']=800
               d['##features#cpu_freq#min']=''
               d['##features#freq#min']=d['##features#gpu_freq#min']

               d['##features#processed#min']='yes'

               # Add throughput (images/second)
               tall=d.get('##characteristics#run#execution_time_classify#all',[]) # It's internal VTA measurements
               if len(tall)>0:
                  tnew=[]
                  for t in tall:
                      t1=1/t
                      tnew.append(t1)
                  
                  r=ck.access({'action':'stat_analysis',
                               'module_uoa':'experiment',
                               'dict':d,
                               'dict1':{'##characteristics#run#inference_throughput':tnew}
                              })
                  if r['return']>0: return r

               # Unify batch size
               batch=1 # for now only 1 is supported in this artifact
               d['##features#batch_size#min']=batch

               # inference latency
               d['##features#measuring_latency#min']='yes'

               r=ck.access({'action':'stat_analysis',
                            'module_uoa':'experiment',
                            'dict':d,
                            'dict1':{'##characteristics#run#inference_latency':tall}
                           })
               if r['return']>0: return r

               r=ck.access({'action':'stat_analysis',
                            'module_uoa':'experiment',
                            'dict':d,
                            'dict1':{'##characteristics#run#prediction_time_avg_s':tall}
                           })
               if r['return']>0: return r

               # Add accuracy (was calculated through separate experiment)
               r=ck.access({'action':'stat_analysis',
                            'module_uoa':'experiment',
                            'dict':d,
                            'dict1':{'##characteristics#run#accuracy_top1':[accuracy_top1]}
                           })
               if r['return']>0: return r

               # Add accuracy (was calculated through separate experiment)
               r=ck.access({'action':'stat_analysis',
                            'module_uoa':'experiment',
                            'dict':d,
                            'dict1':{'##characteristics#run#accuracy_top5':[accuracy_top5]}
                           })
               if r['return']>0: return r

               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
