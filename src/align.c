//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 1100          //>longest entity + longest mention + 1
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TEXT_MODEL "Text Model"
#define KG_MODEL "KG Model"
#define JOINT_MODEL "Joint Model"
#define MAX_MENTION 135

typedef float real;                    // Precision of float numbers

struct anchor_item {
    long long start_pos;
    long long length;
    int entity_index;
};

struct vocab_item {
    long long cn;
    char *item;
};

struct model_var {
    char train_file[MAX_STRING], output_file[MAX_STRING];
    char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
    struct vocab_item *vocab;
    int *vocab_hash;
    long long vocab_max_size, vocab_size;
    long long train_items, item_count_actual, file_size;
    real starting_alpha, alpha;
    real *syn0, *syn1neg;
    char name[MAX_STRING];
    int *table;
}text_model, kg_model;

struct model_var2 {
    char train_file[MAX_STRING];
    long long train_items, item_count_actual, file_size, anchor_count;
    real starting_alpha, alpha;
    char name[MAX_STRING];
}joint_model;

const int vocab_hash_size = 30000000;  // Maximum items in the vocabulary
int local_iter=0, binary = 0, debug_mode = 2, window = 5, min_count = 0, num_threads = 12, min_reduce = 1, cw = 0, sg = 1;
long long layer1_size = 100;
long long iter = 5, tmp_anchor_num = 0;
real alpha = 0.025, sample = 1e-3;
real *expTable;
clock_t start;
char save_vocab_path[MAX_STRING], read_vocab_path[MAX_STRING], output_path[MAX_STRING];
int negative = 5;
const int table_size = 1e8;
const char save_interval = '\t';
struct model_var *p_model = NULL;

void InitUnigramTable() {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    p_model->table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < p_model->vocab_size; a++) train_items_pow += pow(p_model->vocab[a].cn, power);
    i = 0;
    d1 = pow(p_model->vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        p_model->table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(p_model->vocab[i].cn, power) / train_items_pow;
        }
        if (i >= p_model->vocab_size) i = p_model->vocab_size - 1;
    }
}
//return the num of mention words
//split_pos indicates word start position in mention, range from 0 to mention length -1
int SplitMention(int *split_pos, char *mention) {
    int a = 0, b = 0, ch;
    if(mention[b]!=0){
        split_pos[a] = 0;
        a++;
        b++;
    }
    while (mention[b]!=0) {
        ch = mention[b];
        b++;
        if (ch == ' ')
            if(mention[b]!=0){
                split_pos[a] = b;
                a++;
                if(a>=MAX_MENTION){
                    printf("error! anchor's mention length is larger than %d!\n", MAX_MENTION);
                    printf("%s\n",mention);
                    break;
                }
            }
    }
    return a;
}

//return negative if that's not anchor, the int indicates how many words should be offset.
//return the word's start pos
int ReadAnchor(char *item, FILE *fin){
    int a = 0, ch, is_anchor = 0, prev_brace = 0, post_brace = 0, word_num=0, anchor_pos=0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if( prev_brace==0 ){
            if(ch == '['){
                prev_brace = 1;
                continue;
            }
            else{
                a=0;
                is_anchor --;
                break;
            }
        }
        is_anchor--;
        if(ch == ' ') word_num ++;
        if (ch == '\n' || word_num>MAX_MENTION) {
            a = 0;
            break;
        }
        if (ch == '|')
            anchor_pos = a+1;
        if ( (ch == ']') && (post_brace==0)){
            post_brace = 1;
            continue;
        }
        if(post_brace==1){
            if(ch == ']')
                is_anchor = anchor_pos;
            break;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1){printf("error! too long string:\n %s\n",item); a--;break;}
    }
    item[a] = 0;
    return is_anchor;
}

// Reads a single word or anchor from a file, assuming space + tab + EOL to be word boundaries, [[anchor]]
//return -1 if read a word, 0 if [[anchor]], '|''s pos(1-length) if [[entity|mention]]
int ReadItem(char *item, FILE *fin) {
    int a = 0, ch, is_anchor=-1;
    
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == ']') || (ch == '[')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                if (ch == '[') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return -1;
            }
            else if (ch == '['){
                is_anchor = ReadAnchor(item, fin);
                if (is_anchor<0){
                    fseek(fin, is_anchor, SEEK_CUR);
                    continue;
                }
                else return is_anchor;
            }
            else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
    return -1;
}

void ReadEntity(char *item, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ';') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return;
            } else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
}

// Returns hash value of a word
int GetItemHash(char *item) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(item); a++) hash = hash * 257 + item[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (p_model->vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, p_model->vocab[p_model->vocab_hash[hash]].item)) return p_model->vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int SearchTextVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (text_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, text_model.vocab[text_model.vocab_hash[hash]].item)) return text_model.vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int SearchKgVocab(char *item) {
    unsigned int hash = GetItemHash(item);
    while (1) {
        if (kg_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(item, kg_model.vocab[kg_model.vocab_hash[hash]].item)) return kg_model.vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Adds a word or entity to the vocabulary
int AddItemToVocab(char *item) {
    unsigned int hash, length = strlen(item) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    p_model->vocab[p_model->vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(p_model->vocab[p_model->vocab_size].item, item);
    p_model->vocab[p_model->vocab_size].cn = 0;
    p_model->vocab_size++;
    // Reallocate memory if needed
    if (p_model->vocab_size + 2 >= p_model->vocab_max_size) {
        p_model->vocab_max_size += 10000;
        p_model->vocab = (struct vocab_item *)realloc(p_model->vocab, p_model->vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetItemHash(item);
    while (p_model->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    p_model->vocab_hash[hash] = p_model->vocab_size - 1;
    return p_model->vocab_size - 1;
}

// Used later for sorting by item counts
int VocabCompare( const void *a, const void *b) {
    return ((struct vocab_item *)b)->cn - ((struct vocab_item *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&p_model->vocab[1], p_model->vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) p_model->vocab_hash[a] = -1;
    size = p_model->vocab_size;
    p_model->train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((p_model->vocab[a].cn < min_count) && (a != 0)) {
            p_model->vocab_size--;
            free(p_model->vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetItemHash(p_model->vocab[a].item);
            while (p_model->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            p_model->vocab_hash[hash] = a;
            p_model->train_items += p_model->vocab[a].cn;
        }
    }
    p_model->vocab = (struct vocab_item *)realloc(p_model->vocab, (p_model->vocab_size + 1) * sizeof(struct vocab_item));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < p_model->vocab_size; a++) if (p_model->vocab[a].cn > min_reduce) {
        p_model->vocab[b].cn = p_model->vocab[a].cn;
        p_model->vocab[b].item = p_model->vocab[a].item;
        b++;
    } else free(p_model->vocab[a].item);
    p_model->vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) p_model->vocab_hash[a] = -1;
    for (a = 0; a < p_model->vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetItemHash(p_model->vocab[a].item);
        while (p_model->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        p_model->vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void LearnVocabFromTrainFile() {
    char item[MAX_STRING], tmp_item[MAX_STRING];
    FILE *fin;
    long long a, i, b;
    size_t tmp_item_len = 0;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    for (a = 0; a < vocab_hash_size; a++) p_model->vocab_hash[a] = -1;
    fin = fopen(p_model->train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    p_model->vocab_size = 0;
    AddItemToVocab((char *)"</s>");
    while (1) {
        if(!strcmp(TEXT_MODEL, p_model->name))
            anchor_pos = ReadItem(item, fin);
        else
            ReadEntity(item, fin);
        if (feof(fin)) break;
        if (anchor_pos >= 0){
            words_in_mention = SplitMention(word_begin, &item[anchor_pos]);
            tmp_anchor_num ++;
        }
        else
            words_in_mention = 1;
        for(b=0;b<words_in_mention;b++){
            if(anchor_pos==-1){
                tmp_item_len = strlen(item);
                strncpy(tmp_item, item, tmp_item_len);
            }
            else if(b+1<words_in_mention){
                tmp_item_len = word_begin[b+1]-1-word_begin[b];
                strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
            }
            else{
                tmp_item_len = strlen(&item[anchor_pos+word_begin[b]]);
                strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
            }
            tmp_item[tmp_item_len] = 0;
            p_model->train_items++;
            if ((debug_mode > 1) && (p_model->train_items % 100000 == 0)) {
                printf("%lldK%c", p_model->train_items / 1000, 13);
                fflush(stdout);
            }
            i = SearchVocab(tmp_item);
            if (i == -1) {
                a = AddItemToVocab(tmp_item);
                p_model->vocab[a].cn = 1;
            } else p_model->vocab[i].cn++;
            if (p_model->vocab_size > vocab_hash_size * 0.7) ReduceVocab();
        }
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("%s Vocab size: %lld\n", p_model->name, p_model->vocab_size);
        printf("Items of %s in train file: %lld\n", p_model->name, p_model->train_items);
    }
    p_model->file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    long long i;
    FILE *fo = fopen(p_model->save_vocab_file, "wb");
    for (i = 0; i < p_model->vocab_size; i++) fprintf(fo, "%s %lld\n", p_model->vocab[i].item, p_model->vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c;
    char item[MAX_STRING];
    FILE *fin = fopen(p_model->read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) p_model->vocab_hash[a] = -1;
    p_model->vocab_size = 0;
    while (1) {
        ReadEntity(item, fin);
        if (feof(fin)) break;
        a = AddItemToVocab(item);
        fscanf(fin, "%lld%c", &p_model->vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("%s Vocab size: %lld\n", p_model->name, p_model->vocab_size);
        printf("Items of %s in train file: %lld\n", p_model->name, p_model->train_items);
    }
    fin = fopen(p_model->train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    p_model->file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&(p_model->syn0), 128, (long long)p_model->vocab_size * layer1_size * sizeof(real));
    if (p_model->syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (negative>0) {
        a = posix_memalign((void **)&(p_model->syn1neg), 128, (long long)p_model->vocab_size * layer1_size * sizeof(real));
        if (p_model->syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < p_model->vocab_size; a++) for (b = 0; b < layer1_size; b++)
            p_model->syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < p_model->vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        p_model->syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}

void *TrainTextModelThread(void *id) {
    long long a, b, d, word=-1, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    char item[MAX_STRING], tmp_item[MAX_STRING];
    size_t tmp_item_len = 0;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(text_model.train_file, "rb");
    fseek(fi, text_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    while (1) {
        if (word_count - last_word_count > 10000) {
            text_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, text_model.name, text_model.alpha,
                       text_model.item_count_actual / (real)(text_model.train_items + 1) * 100,
                       text_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            text_model.alpha = text_model.starting_alpha * (1 - text_model.item_count_actual / (real)(iter * text_model.train_items + 1));
            if (text_model.alpha < text_model.starting_alpha * 0.0001) text_model.alpha = text_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                anchor_pos = ReadItem(item, fi);
                if (feof(fi)) break;
                if (anchor_pos >= 0)
                    words_in_mention = SplitMention(word_begin, &item[anchor_pos]);
                else
                    words_in_mention = 1;
                for(b=0;b<words_in_mention;b++){
                    if(anchor_pos==-1){
                        tmp_item_len = strlen(item);
                        strncpy(tmp_item, item, tmp_item_len);
                    }
                    else if(b+1<words_in_mention){
                        tmp_item_len = word_begin[b+1]-1-word_begin[b];
                        strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    else{
                        tmp_item_len = strlen(&item[anchor_pos+word_begin[b]]);
                        strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    tmp_item[tmp_item_len] = 0;
                    word = SearchTextVocab(tmp_item);
                    if (word == -1) continue;
                    word_count++;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ran = (sqrt(text_model.vocab[word].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                if(word == 0 || sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > text_model.train_items / num_threads)) {
            text_model.item_count_actual += word_count - last_word_count;
            break;
        }
        
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        //train skip-gram
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            l1 = last_word * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = text_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += text_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * text_model.alpha;
                else if (f < -MAX_EXP) g = (label - 0) * text_model.alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * text_model.alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * text_model.syn0[c + l1];
            }
            // Learn weights input -> hidden
            for (c = 0; c < layer1_size; c++) text_model.syn0[c + l1] += neu1e[c];
        }
        
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void *TrainKgModelThread(void *id) {
    long long d, entity, head_entity = -1, line_entity_count = 0, is_read_head = 1, sentence_length = 0, sentence_position = 0;
    long long entity_count = 0, last_entity_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    char tmp_item[MAX_STRING];
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(kg_model.train_file, "rb");
    fseek(fi, kg_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    
    while(1){
        ReadEntity(tmp_item, fi);
        entity = SearchKgVocab(tmp_item);
        entity_count ++;
        if (feof(fi) || (entity==0)) break;
    }
    head_entity = -1;
    is_read_head = 1;
    line_entity_count = 0;
    while (1) {
        if (entity_count - last_entity_count > 10000) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            last_entity_count = entity_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  entities/thread/sec: %.2fk  ", 13, kg_model.name, kg_model.alpha,
                       kg_model.item_count_actual / (real)(kg_model.train_items + 1) * 100,
                       kg_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            kg_model.alpha = kg_model.starting_alpha * (1 - kg_model.item_count_actual / (real)(iter * kg_model.train_items + 1));
            if (kg_model.alpha < kg_model.starting_alpha * 0.0001) kg_model.alpha = kg_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while(1){
                ReadEntity(tmp_item, fi);
                if (feof(fi)) break;
                entity_count ++;
                line_entity_count++;
                if(is_read_head==1){
                    head_entity = SearchKgVocab(tmp_item);
                    if(head_entity==0) line_entity_count = 0;
                    if(head_entity>0 && line_entity_count==1) is_read_head=0;
                    else head_entity = -1;
                    continue;
                }
                else if(head_entity!=-1){
                    entity = SearchKgVocab(tmp_item);
                    if (entity == -1) continue;
                    if (entity == 0) {
                        line_entity_count = 0;
                        is_read_head=1;
                        break;
                    }
                }
                else {is_read_head=1;continue;}
                sen[sentence_length] = entity;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (entity_count > kg_model.train_items / num_threads)) {
            kg_model.item_count_actual += entity_count - last_entity_count;
            break;
        }
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        //train skip-gram
        for (; sentence_position<sentence_length; sentence_position++){
            entity = sen[sentence_position];
            if (entity == -1) continue;
            l1 = entity * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = head_entity;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = kg_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (kg_model.vocab_size - 1) + 1;
                    if (target == head_entity) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * kg_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * kg_model.alpha;
                else if (f < -MAX_EXP) g = (label - 0) * kg_model.alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * kg_model.alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * kg_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) kg_model.syn1neg[c + l2] += g * kg_model.syn0[c + l1];
            }
            // Learn weights input -> hidden
            for (c = 0; c < layer1_size; c++) kg_model.syn0[c + l1] += neu1e[c];
        }
        sentence_length = 0;
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

//use entity to predict word
void *TrainJointModelThread(void *id) {
    long long a, b, d, word=-1, last_word, sentence_length = 0, sentence_position = 0, anchor_position = 0;
    long long word_count = 0, anchor_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    char tmp_item[MAX_STRING], item[MAX_STRING];
    size_t tmp_item_len = 0;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1;
    real f, g;
    clock_t now;
    struct anchor_item *anchors = (struct anchor_item *)calloc(MAX_SENTENCE_LENGTH, sizeof(struct anchor_item));
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(joint_model.train_file, "rb");
    fseek(fi, joint_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    while (1) {
        if (word_count - last_word_count > 10000) {
            joint_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, joint_model.name, joint_model.alpha,
                       joint_model.item_count_actual / (real)(joint_model.train_items + 1) * 100,
                       joint_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            joint_model.alpha = joint_model.starting_alpha * (1 - joint_model.item_count_actual / (real)(iter * joint_model.train_items + 1));
            if (joint_model.alpha < joint_model.starting_alpha * 0.0001) joint_model.alpha = joint_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            anchor_count = 0;
            while(1){
                anchor_pos = ReadItem(item, fi);
                if (feof(fi)) break;
                if (anchor_pos >= 0){
                    words_in_mention = SplitMention(word_begin, &item[anchor_pos]);
                    anchors[anchor_count].start_pos = sentence_length;
                }
                else
                    words_in_mention = 1;
                for(b=0;b<words_in_mention;b++){
                    if(anchor_pos==-1){
                        tmp_item_len = strlen(item);
                        strncpy(tmp_item, item, tmp_item_len);
                    }
                    else if(b+1<words_in_mention){
                        tmp_item_len = word_begin[b+1]-1-word_begin[b];
                        strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    else{
                        tmp_item_len = strlen(&item[anchor_pos+word_begin[b]]);
                        strncpy(tmp_item, &item[anchor_pos+word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    tmp_item[tmp_item_len] = 0;
                    word = SearchTextVocab(tmp_item);
                    if (word == -1) continue;
                    word_count++;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0 && anchor_pos==-1) {
                        real ran = (sqrt(text_model.vocab[word].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                if(anchor_pos>=0 && b >= words_in_mention-1){
                    anchors[anchor_count].length = sentence_length - anchors[anchor_count].start_pos;
                    if(anchors[anchor_count].length>0){
                        if (anchor_pos==0)
                            tmp_item_len = strlen(item);
                        else
                            tmp_item_len = anchor_pos-1;
                        strncpy(tmp_item, item, tmp_item_len);
                        tmp_item[tmp_item_len] = 0;
                        anchors[anchor_count].entity_index = SearchKgVocab(tmp_item);
                        if(anchors[anchor_count].entity_index!=-1){
                            anchor_count++;
                            if(anchor_count >= MAX_SENTENCE_LENGTH) anchor_count--;
                        }
                    }
                }
                if(word == 0 || sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > joint_model.train_items / num_threads)) {
            joint_model.item_count_actual += word_count - last_word_count;
            break;
        }
        
        if(anchor_count<=0){
            sentence_length = 0;
            continue;
        }
        if(sg){
            for(anchor_position=0;anchor_position<anchor_count;anchor_position++){
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                b = next_random % window;
                sentence_position = anchors[anchor_position].start_pos;
                l1 = anchors[anchor_position].entity_index * layer1_size;
                //train skip-gram
                for (a = b; a < window * 2 + 1 - b; a++)
                    if(a == window)
                        sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                    else {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        // NEGATIVE SAMPLING
                        if (negative > 0) for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = last_word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = text_model.table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                                if (target == last_word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += kg_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * joint_model.alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * joint_model.alpha;
                            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * joint_model.alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * kg_model.syn0[c + l1];
                        }
                        // Learn weights input -> hidden
                        for (c = 0; c < layer1_size; c++) kg_model.syn0[c + l1] += neu1e[c];
                    }
            }
        }// end of sg


        if(cw){
            for(anchor_position=0;anchor_position<anchor_count;anchor_position++){
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                b = next_random % window;
                sentence_position = anchors[anchor_position].start_pos;
                word = anchors[anchor_position].entity_index;
                if (word == -1) continue;
                //train skip-gram
                for (a = b; a < window * 2 + 1 - b; a++)
                    if(a == window)
                        sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                    else {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        l1 = last_word * layer1_size;
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        // NEGATIVE SAMPLING
                        if (negative > 0) for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long)25214903917 + 11;
                                target = kg_model.table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (kg_model.vocab_size - 1) + 1;
                                if (target == word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += text_model.syn0[c + l1] * kg_model.syn1neg[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * joint_model.alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * joint_model.alpha;
                            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * joint_model.alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * kg_model.syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++) kg_model.syn1neg[c + l2] += g * text_model.syn0[c + l1];
                        }
                        // Learn weights input -> hidden
                        for (c = 0; c < layer1_size; c++) text_model.syn0[c + l1] += neu1e[c];
                    }
            }
        }// end of cw
        
        sentence_length = 0;
    }
    fclose(fi);
    free(anchors);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainTextModel(){
    long a;
    p_model = &text_model;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training using file %s\n", text_model.train_file);
    text_model.item_count_actual = 0;
    text_model.starting_alpha = text_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainTextModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    p_model = NULL;
}

void TrainKgModel(){
    long a;
    p_model = &kg_model;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training using file %s\n", kg_model.train_file);
    kg_model.item_count_actual = 0;
    kg_model.starting_alpha = kg_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainKgModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    p_model = NULL;
}

void TrainJointModel() {
    long a;
    p_model = NULL;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    start = clock();
    printf("\nStarting training using file %s\n", joint_model.train_file);
    joint_model.item_count_actual = 0;
    joint_model.starting_alpha = joint_model.alpha;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainJointModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
}

void SaveVector(int id, char *postfix){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b;
    sprintf(tmp_output_file, "%s%d%s", p_model->output_file, id, postfix);
    fo = fopen(tmp_output_file, "wb");
    // Save the item vectors
    fprintf(fo, "%lld %lld\n", p_model->vocab_size, layer1_size);
    for (a = 0; a < p_model->vocab_size; a++) {
        fprintf(fo, "%s%c", p_model->vocab[a].item, save_interval);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(p_model->syn0[a * layer1_size + b]), sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", p_model->syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void SaveWordVector(int id, char *postfix){
    p_model = &text_model;
    SaveVector(id, postfix);
}

void SaveEntityVector(int id, char *postfix){
    p_model = &kg_model;
    SaveVector(id, postfix);
}

void InitModel(){
    if(!strcmp(TEXT_MODEL, p_model->name))
        p_model->vocab_max_size = 2500000;      //vocab word size is 2.7m
    else
        p_model->vocab_max_size = 5000000;      //vocab entity size is 5.13m
    p_model->vocab_size = 0;
    p_model->train_items = 0;
    p_model->item_count_actual = 0;
    p_model->file_size = 0;
    p_model->alpha = alpha;
    p_model->vocab = (struct vocab_item *)calloc(p_model->vocab_max_size, sizeof(struct vocab_item));
    p_model->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    if (read_vocab_path[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
    if (save_vocab_path[0] != 0) SaveVocab();
    if (output_path[0] == 0) return;
    InitNet();
    if (negative > 0) InitUnigramTable();
}

void InitTextModel(){
    p_model = &text_model;
    InitModel();
    p_model = NULL;
}

void InitKgModel(){
    p_model = &kg_model;
    min_count = 1;
    InitModel();
    p_model = NULL;
}

void InitJointModel(){
    joint_model.file_size = text_model.file_size;
    joint_model.train_items = text_model.train_items;
    joint_model.item_count_actual = 0;
    joint_model.alpha = alpha;
    joint_model.anchor_count = tmp_anchor_num;
    printf("Total anchors in train file: %lld\n", joint_model.anchor_count);
}


int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Joint word&entity VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train_text <file>\n");
        printf("\t\tUse text data from <file> to train the text model\n");
        printf("\t-train_kg <file>\n");
        printf("\t\tUse knowledge data from <file> to train the knowledge model\n");
        printf("\t-train_anchor <file>\n");
        printf("\t\tUse anchor data from <file> to train align model\n");
        printf("\t-output_path <file>\n");
        printf("\t\tUse <path> to save the resulting vectors / vocab\n");
        printf("\t-read_vocab_path <file>\n");
        printf("\t\tUse <path> to read the word and entity vocab\n");
        printf("\t-save_vocab_path <file>\n");
        printf("\t\tUse <path> to save the word and entity vocab\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word  / entity vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between (anchor) words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of (anchor) words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-cw <int>\n");
        printf("\t\talign word vector to entity; default is 0\n");
        printf("\t-sg <int>\n");
        printf("\t\talign entity vector to word; default is 1\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    
    read_vocab_path[0]=0;
    save_vocab_path[0]=0;
    output_path[0]=0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train_text", argc, argv)) > 0) strcpy(text_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_kg", argc, argv)) > 0) strcpy(kg_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-train_anchor", argc, argv)) > 0) strcpy(joint_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-output_path", argc, argv)) > 0) strcpy(output_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_vocab_path", argc, argv)) > 0) strcpy(read_vocab_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-save_vocab_path", argc, argv)) > 0) strcpy(save_vocab_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-cw", argc, argv)) > 0) cw = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sg", argc, argv)) > 0) sg = atoi(argv[i + 1]);
    
    if(save_vocab_path[0]!=0){
        sprintf(text_model.save_vocab_file, "%svocab_word.txt", save_vocab_path);
        sprintf(kg_model.save_vocab_file, "%svocab_entity.txt", save_vocab_path);
    }
    if(output_path[0]!=0){
        sprintf(text_model.output_file, "%svectors_word", output_path);
        sprintf(kg_model.output_file, "%svectors_entity", output_path);
    }
    if(read_vocab_path[0]!=0){
        sprintf(text_model.read_vocab_file, "%svocab_word.txt", read_vocab_path);
        sprintf(kg_model.read_vocab_file, "%svocab_entity.txt", read_vocab_path);
    }
    
    strcpy(text_model.name, (char *)TEXT_MODEL);
    strcpy(kg_model.name, (char *)KG_MODEL);
    strcpy(joint_model.name, (char *)JOINT_MODEL);
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    //read vocab & initilize text model and kg model
    InitTextModel();
    InitKgModel();
    InitJointModel();
    //start training
    local_iter = 0;
    while(local_iter<iter){
        local_iter++;
        printf("Start training the %d time... ", local_iter);
        TrainTextModel();
        TrainKgModel();
        TrainJointModel();
        printf("\niter %d success!\n", local_iter);
        if(local_iter%1==0){
            printf("saving results...\n");
            SaveWordVector(local_iter, ".dat");
            SaveEntityVector(local_iter, ".dat");
        }
    }
    return 0;
}
