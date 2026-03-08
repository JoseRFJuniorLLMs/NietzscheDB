import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useState,
    type ReactNode,
} from "react"
import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

type Theme = "dark" | "light"

interface ThemeContextValue {
    theme: Theme
    toggleTheme: () => void
    setTheme: (t: Theme) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

const STORAGE_KEY = "nietzsche_theme"

function applyTheme(theme: Theme) {
    const root = document.documentElement
    root.style.transition = "background-color 0.3s ease, color 0.3s ease"
    if (theme === "dark") {
        root.classList.add("dark")
    } else {
        root.classList.remove("dark")
    }
}

function getInitialTheme(): Theme {
    if (typeof window === "undefined") return "dark"
    try {
        const stored = localStorage.getItem(STORAGE_KEY)
        if (stored === "light" || stored === "dark") return stored
    } catch {
        // localStorage unavailable
    }
    return "dark"
}

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [theme, setThemeState] = useState<Theme>(getInitialTheme)

    useEffect(() => {
        applyTheme(theme)
    }, [theme])

    const setTheme = useCallback((t: Theme) => {
        setThemeState(t)
        try {
            localStorage.setItem(STORAGE_KEY, t)
        } catch {
            // ignore
        }
    }, [])

    const toggleTheme = useCallback(() => {
        setTheme(theme === "dark" ? "light" : "dark")
    }, [theme, setTheme])

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useThemeMode() {
    const ctx = useContext(ThemeContext)
    if (!ctx) throw new Error("useThemeMode must be used within ThemeProvider")
    return ctx
}

export function ThemeToggle({ className }: { className?: string }) {
    const { theme, toggleTheme } = useThemeMode()

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className={cn(
                "h-8 w-8 text-zinc-400 transition-colors hover:text-white",
                className
            )}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
            {theme === "dark" ? (
                <Sun className="h-4 w-4" />
            ) : (
                <Moon className="h-4 w-4" />
            )}
        </Button>
    )
}
