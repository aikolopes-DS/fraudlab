param([Parameter(Mandatory=$true)][string]$RequestPath)
$ErrorActionPreference = 'Stop'
$taskSpeechRequest = Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskSpeaker = New-Object -ComObject SAPI.SpVoice
$taskSelectedVoice = $null
foreach ($taskCandidateVoice in $taskSpeaker.GetVoices()) {
    if ($taskCandidateVoice.GetDescription().StartsWith($taskSpeechRequest.voice)) {
        $taskSelectedVoice = $taskCandidateVoice
        break
    }
}
if ($null -eq $taskSelectedVoice) { throw "Voz solicitada nao instalada: $($taskSpeechRequest.voice)" }
$taskSpeaker.Voice = $taskSelectedVoice
$taskSpeaker.Rate = [int]$taskSpeechRequest.rate
foreach ($taskUtterance in $taskSpeechRequest.utterances) {
    $taskAudioStream = New-Object -ComObject SAPI.SpFileStream
    # SAFT22kHz16BitMono, PCM. CreateForWrite is 3.
    $taskAudioStream.Format.Type = 22
    $taskAudioStream.Open($taskUtterance.path, 3, $false)
    try {
        $taskSpeaker.AudioOutputStream = $taskAudioStream
        [void]$taskSpeaker.Speak([string]$taskUtterance.text, 0)
    } finally {
        $taskAudioStream.Close()
    }
}
Write-Output "Narracao gerada: $($taskSpeechRequest.utterances.Count) trechos."
