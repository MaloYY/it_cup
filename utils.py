import json

def fine_print(scores):
    print("m_acc:    ", scores['mean_test_accuracy'])
    print("m_f1:     ", scores['mean_test_f1'])
    print("m_rauc:   ", scores['mean_test_roc_auc'])
    print("m_prec:   ", scores['mean_test_precision'])
    print("m_recall: ", scores['mean_test_recall'])


def save_config(id, config, model):
    with open('log_config.txt', 'a') as outfile:
        outfile.write(f"ID: p{id}  MODEL: {model}\n")
        json.dump(config, outfile)
        outfile.write('\n================\n\n')

    