import synapseclient
import synapseutils


if __name__ == "__main__":
    syn = synapseclient.Synapse()
    syn.login('shaharabany', 't6gTghbn')
    files = synapseutils.syncFromSynapse(syn, ' syn3193805 ')