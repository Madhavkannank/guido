import { useState, useEffect, useRef, useMemo } from 'react'

const WORDS = [
  'guido',
  '\u0917\u0941\u0907\u0921\u094B',
  '\u0B95\u0BC1\u0BAF\u0BCD\u0B9F\u0BCB',
]

const TYPING_SPEED   = 500
const DELETING_SPEED = 80
const PAUSE_AFTER    = 1000
const PAUSE_BEFORE   = 400

function getGraphemes(str) {
  try {
    const seg = new Intl.Segmenter('en', { granularity: 'grapheme' })
    return [...seg.segment(str)].map(s => s.segment)
  } catch {
    return [str]
  }
}

export default function TypingTitle() {
  const allChars = useMemo(() => WORDS.map(w => getGraphemes(w)), [])
  const [wordIdx, setWordIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [phase, setPhase] = useState('typing') // 'typing' | 'pausing' | 'deleting' | 'waiting'
  const timer = useRef(null)

  useEffect(() => {
    const chars = allChars[wordIdx]

    switch (phase) {
      case 'typing':
        if (charIdx < chars.length) {
          timer.current = setTimeout(() => setCharIdx(c => c + 1), TYPING_SPEED)
        } else {
          setPhase('pausing')
        }
        break
      case 'pausing':
        timer.current = setTimeout(() => setPhase('deleting'), PAUSE_AFTER)
        break
      case 'deleting':
        if (charIdx > 0) {
          timer.current = setTimeout(() => setCharIdx(c => c - 1), DELETING_SPEED)
        } else {
          setPhase('waiting')
        }
        break
      case 'waiting':
        timer.current = setTimeout(() => {
          setWordIdx(w => (w + 1) % WORDS.length)
          setPhase('typing')
        }, PAUSE_BEFORE)
        break
    }

    return () => clearTimeout(timer.current)
  }, [charIdx, phase, wordIdx, allChars])

  const display = allChars[wordIdx].slice(0, charIdx).join('')

  return (
    <span className="typing-text">
      {display}
      <span className="typing-cursor">|</span>
    </span>
  )
}
