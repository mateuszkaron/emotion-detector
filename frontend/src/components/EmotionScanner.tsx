import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Camera, Activity, Shield, Scan, AlertTriangle } from "lucide-react"

export default function Component() {
  const [isScanning, setIsScanning] = useState(false)
  const [scanProgress, setScanProgress] = useState(0)
  const [currentEmotion, setCurrentEmotion] = useState("NEUTRAL")
  const [confidence, setConfidence] = useState(0)
  const [accessGranted, setAccessGranted] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [allConfidences, setAllConfidences] = useState<{[key: string]: number}>({})
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const emotions = ["ANALYZING", "ANGRY","DISGUSTED", "FEARFUL", "HAPPY", "NEUTRAL", "SAD", "SURPRISED"]
  const scanData = [
    { label: "BIOMETRIC ID", value: "USER_7849" },
    { label: "SCAN TYPE", value: "EMOTION_DEEP" },
    { label: "SECURITY LVL", value: "ALPHA-4" },
    { label: "TIMESTAMP", value: new Date().toLocaleTimeString() },
  ]

  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: true,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }

        const audioContext = new AudioContext()
        const analyser = audioContext.createAnalyser()
        const microphone = audioContext.createMediaStreamSource(stream)
        microphone.connect(analyser)

        const dataArray = new Uint8Array(analyser.frequencyBinCount)

        const updateAudioLevel = () => {
          analyser.getByteFrequencyData(dataArray)
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length
          setAudioLevel(average / 255)
          requestAnimationFrame(updateAudioLevel)
        }
        updateAudioLevel()
      } catch (error) {
        console.error("Camera access denied:", error)
      }
    }

    initCamera()
  }, [])

    // Wysylanie klatki do API
  const sendFrameToApi = async () => {
    if (!videoRef.current) return

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas')
      canvasRef.current.width = 640
      canvasRef.current.height = 480
    }
    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    ctx.drawImage(videoRef.current, 0, 0, 640, 480)

    const blob: Blob | null = await new Promise(resolve =>
      canvasRef.current!.toBlob(resolve, 'image/jpeg')
    )
    if (!blob) return

    const formData = new FormData()
    formData.append('image', blob, 'snapshot.jpg')

    try {
      videoRef.current.pause()

      // Reset progress
      setScanProgress(0)

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (response.ok && data.emotion) {
        setCurrentEmotion(data.emotion.toUpperCase())
        setConfidence(data.confidence ?? 100)
        setAllConfidences(data.all_confidences ?? {})
      } else {
        setCurrentEmotion("NO FACE DETECTED")
        setConfidence(0)
        setAllConfidences({})
      }
    } catch (e) {
      setCurrentEmotion("ERROR")
      setConfidence(0)
    }

    // Symuluj progres skanowania od 0 do 100% w ~2 sekundy
    let progress = 0
    const interval = setInterval(() => {
      progress += 5 // co 100ms zwiększ o 5%
      if (progress >= 100) {
        progress = 100
        clearInterval(interval)

        // Odmrażaj video po osiągnięciu 100%
        videoRef.current?.play()
      }
      setScanProgress(progress)
    }, 100)
  }


  useEffect(() => {
    if (isScanning) {
      const interval = setInterval(() => {
        setScanProgress((prev) => {
          if (prev >= 100) {
            setIsScanning(false)
            setAccessGranted(true)
            setTimeout(() => setAccessGranted(false), 3000)
            return 0
          }
          return prev + 2
        })
      }, 100)

      sendFrameToApi()

      return () => clearInterval(interval)
    }
  }, [isScanning])

  const startScan = () => {
    setIsScanning(true)
    setScanProgress(0)
    setAccessGranted(false)
    setCurrentEmotion("ANALYZING")
    setConfidence(0)
  }

  return (
    <div className="min-h-screen bg-black text-green-400 font-mono overflow-hidden relative">
      {/* Animated Background Grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-black"></div>
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: "50px 50px",
          }}
        ></div>
      </div>

      {/* Main Interface */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4">
        {/* Header */}
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <h1 className="text-4xl md:text-6xl font-bold text-cyan-400 mb-2 tracking-wider">EMOTION SCANNER</h1>
          <p className="text-green-400 text-lg tracking-widest">BIOMETRIC SECURITY PROTOCOL v2.7.9</p>
        </motion.div>

        {/* Main Scanner Interface */}
        <div className="relative max-w-4xl w-full">
          {/* Central Video Feed */}
          <div className="relative mx-auto w-fit">
            {/* Video Container with CRT Effect */}
            <div className="relative bg-black p-4 rounded-lg border-2 border-cyan-500 shadow-2xl shadow-cyan-500/50">
              {/* Video Element */}
              <div className="relative w-[640px] h-[480px] max-w-full bg-black rounded overflow-hidden">
                <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />

                {/* CRT Scanlines Effect */}
                <div
                  className="absolute inset-0 pointer-events-none opacity-30"
                  style={{
                    background: `repeating-linear-gradient(
                      0deg,
                      transparent,
                      transparent 2px,
                      rgba(0, 255, 255, 0.1) 2px,
                      rgba(0, 255, 255, 0.1) 4px
                    )`,
                  }}
                ></div>

                {/* Screen Flicker */}
                <motion.div
                  className="absolute inset-0 bg-cyan-400 mix-blend-overlay pointer-events-none"
                  animate={{ opacity: [0, 0.05, 0] }}
                  transition={{ duration: 0.1, repeat: Number.POSITIVE_INFINITY, repeatDelay: Math.random() * 3 }}
                ></motion.div>

                {/* Blue Glow */}
                <div className="absolute inset-0 bg-blue-500/10 pointer-events-none"></div>

                {/* Facial Recognition Overlay */}
                <AnimatePresence>
                  {isScanning && (
                    <motion.div
                      className="absolute inset-0 pointer-events-none"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      {/* Face Detection Rectangle */}
                      <motion.div
                        className="absolute border-2 border-red-500"
                        style={{
                          left: "25%",
                          top: "20%",
                          width: "50%",
                          height: "60%",
                        }}
                        animate={{
                          borderColor: ["#ef4444", "#22c55e", "#ef4444"],
                          scale: [1, 1.02, 1],
                        }}
                        transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                      >
                        {/* Corner Brackets */}
                        <div className="absolute -top-1 -left-1 w-6 h-6 border-t-4 border-l-4 border-cyan-400"></div>
                        <div className="absolute -top-1 -right-1 w-6 h-6 border-t-4 border-r-4 border-cyan-400"></div>
                        <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-4 border-l-4 border-cyan-400"></div>
                        <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-4 border-r-4 border-cyan-400"></div>
                      </motion.div>

                      {/* Scanning Line */}
                      <motion.div
                        className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-red-500 to-transparent"
                        animate={{ top: ["0%", "100%"] }}
                        transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
                      ></motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Access Granted Overlay */}
                <AnimatePresence>
                  {accessGranted && (
                    <motion.div
                      className="absolute inset-0 bg-green-500/20 flex items-center justify-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <motion.div
                        className="text-green-400 text-4xl font-bold tracking-wider"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 200 }}
                      >
                        ACCESS GRANTED
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Rotating Reticles */}
            <motion.div
              className="absolute -inset-8 pointer-events-none"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
            >
              <div className="absolute top-0 left-1/2 w-8 h-8 -translate-x-1/2 -translate-y-4">
                <Scan className="w-full h-full text-cyan-400" />
              </div>
              <div className="absolute bottom-0 left-1/2 w-8 h-8 -translate-x-1/2 translate-y-4 rotate-180">
                <Scan className="w-full h-full text-cyan-400" />
              </div>
              <div className="absolute left-0 top-1/2 w-8 h-8 -translate-y-1/2 -translate-x-4 -rotate-90">
                <Scan className="w-full h-full text-cyan-400" />
              </div>
              <div className="absolute right-0 top-1/2 w-8 h-8 -translate-y-1/2 translate-x-4 rotate-90">
                <Scan className="w-full h-full text-cyan-400" />
              </div>
            </motion.div>
          </div>

          {/* Side Panels */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
            {/* Left Panel - Scan Data */}
            <motion.div
              className="bg-black/80 border border-cyan-500 rounded-lg p-4 backdrop-blur-sm"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <h3 className="text-cyan-400 text-lg font-bold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                SCAN DATA
              </h3>
              <div className="space-y-2">
                {scanData.map((item, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span className="text-gray-400">{item.label}:</span>
                    <span className="text-green-400 font-mono">{item.value}</span>
                  </div>
                ))}
              </div>

              {/* Progress Bar */}
              <div className="mt-4">
                <div className="flex justify-between text-xs mb-1">
                  <span>SCAN PROGRESS</span>
                  <span>{scanProgress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <motion.div
                    className="bg-gradient-to-r from-cyan-500 to-green-500 h-2 rounded-full"
                    style={{ width: `${scanProgress}%` }}
                    transition={{ duration: 0.1 }}
                  ></motion.div>
                </div>
              </div>
            </motion.div>

            {/* Center Panel - Controls */}
            <motion.div
              className="bg-black/80 border border-cyan-500 rounded-lg p-4 backdrop-blur-sm"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              <h3 className="text-cyan-400 text-lg font-bold mb-4 flex items-center gap-2">
                <Camera className="w-5 h-5" />
                CONTROLS
              </h3>

              <button
                onClick={startScan}
                disabled={isScanning}
                className={`w-full py-3 px-4 rounded-lg font-bold tracking-wider transition-all ${
                  isScanning
                    ? "bg-red-900 text-red-400 cursor-not-allowed"
                    : "bg-green-900 text-green-400 hover:bg-green-800 active:scale-95"
                }`}
              >
                {isScanning ? "SCANNING..." : "INITIATE SCAN"}
              </button>

              {/* Emotion Display */}
              <div className="mt-4 text-center">
                <div className="text-xs text-gray-400 mb-1">DETECTED EMOTION</div>
                <motion.div
                  className="text-xl font-bold text-yellow-400"
                  key={currentEmotion}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {currentEmotion}
                </motion.div>
                <div className="text-xs text-gray-400 mt-1">CONFIDENCE: {confidence.toFixed(1)}%</div>
                {/* Nowy blok: tabela pewności wszystkich emocji */}
                {allConfidences && Object.keys(allConfidences).length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-gray-400 mb-1">ALL EMOTION CONFIDENCES</div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      {Object.entries(allConfidences).map(([emo, conf]) => (
                        <div key={emo} className="flex justify-between">
                          <span className="text-cyan-300">{emo.toUpperCase()}</span>
                          <span className="text-green-300">{conf.toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Right Panel - Audio & Status */}
            <motion.div
              className="bg-black/80 border border-cyan-500 rounded-lg p-4 backdrop-blur-sm"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.9 }}
            >
              <h3 className="text-cyan-400 text-lg font-bold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                AUDIO LEVELS
              </h3>

              {/* Audio Waveform */}
              <div className="flex items-end justify-center gap-1 h-20 mb-4">
                {Array.from({ length: 20 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="bg-green-500 w-2 rounded-t"
                    animate={{
                      height: `${Math.max(10, audioLevel * 100 + Math.random() * 20)}%`,
                    }}
                    transition={{ duration: 0.1 }}
                  ></motion.div>
                ))}
              </div>

              {/* Status Indicators */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs">CAMERA</span>
                  <motion.div
                    className="w-3 h-3 rounded-full bg-green-500"
                    animate={{ opacity: [1, 0.5, 1] }}
                    transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
                  ></motion.div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs">AUDIO</span>
                  <motion.div
                    className="w-3 h-3 rounded-full bg-green-500"
                    animate={{ opacity: [1, 0.5, 1] }}
                    transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
                  ></motion.div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs">NEURAL NET</span>
                  <motion.div
                    className="w-3 h-3 rounded-full bg-cyan-500"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY }}
                  ></motion.div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Footer Warning */}
        <motion.div
          className="mt-8 flex items-center gap-2 text-red-400 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          <AlertTriangle className="w-4 h-4" />
          <span>UNAUTHORIZED ACCESS WILL BE REPORTED TO SECURITY</span>
          <AlertTriangle className="w-4 h-4" />
        </motion.div>
      </div>

      {/* Ambient Lighting Effects */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
        <motion.div
          className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-radial from-cyan-500/10 to-transparent"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.1, 0.3],
          }}
          transition={{ duration: 4, repeat: Number.POSITIVE_INFINITY }}
        ></motion.div>
      </div>
    </div>
  )
}
