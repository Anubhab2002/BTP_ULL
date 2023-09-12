wget https://github.com/yq-wen/overlapping-datasets/releases/download/v0.1/cleaned_dd.zip
unzip -o cleaned_dd.zip
mkdir -p data/ijcnlp_dailydialog_cc/train/
mkdir -p data/ijcnlp_dailydialog_cc/validation/
mkdir -p data/ijcnlp_dailydialog_cc/test/
mv cleaned_dd/dialogs/train_dialogs.txt data/ijcnlp_dailydialog_cc/train/dialogues_train.txt
mv cleaned_dd/dialogs/valid_dialogs.txt data/ijcnlp_dailydialog_cc/validation/dialogues_validation.txt
mv cleaned_dd/dialogs/test_dialogs.txt data/ijcnlp_dailydialog_cc/test/dialogues_test.txt
rm -rf cleaned_dd
rm cleaned_dd.zip
