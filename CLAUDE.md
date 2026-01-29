# Normas para Claude Code

## Git Commits

### PROHIBIDO
- **NUNCA** usar `git commit --amend` sin que el usuario lo pida explícitamente
- **NUNCA** usar `git reset --hard` sin que el usuario lo pida explícitamente
- **NUNCA** usar `git push --force` sin que el usuario lo pida explícitamente
- **NUNCA** usar `--no-verify` para saltarse hooks

### OBLIGATORIO
- Siempre crear commits NUEVOS (no amend)
- Usar formato HEREDOC para mensajes de commit multilinea
- Preguntar al usuario antes de hacer push
- **NUNCA** incluir `Co-Authored-By` en los mensajes de commit
- **NUNCA** usar milestones (h01, h12, etc.) en el scope del commit

### Formato de commit
```bash
git commit -m "$(cat <<'EOF'
tipo: descripción corta

Detalles si son necesarios.
EOF
)"
```

### Tipos de commit
- `feat:` - nueva funcionalidad
- `feat(scope):` - nueva funcionalidad con scope (ej: `feat(matcher):`, `feat(pipeline):`)
- `fix:` - corrección de bug
- `docs:` - documentación
- `refactor:` - refactorización sin cambiar funcionalidad
- `test:` - tests
- `chore:` - tareas de mantenimiento

### Scope válidos (ejemplos)
- `matcher`, `extractor`, `detector`, `pipeline`, `imu`, `mapper`
- **NO usar:** `h01`, `h12`, `aria`, números de milestone

## Metodología de enseñanza

Este proyecto es de **aprendizaje guiado**:
1. Claude pregunta, el usuario responde
2. El usuario escribe el código cuando es posible
3. Claude guía paso a paso, no hace todo de golpe

## Idioma

Comunicar en **español** con el usuario.
